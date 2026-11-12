#include "engine/runtime/serve.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <utility>

#include "engine/runtime/device_buffer.h"
#include "engine/runtime/nvtx.h"
#include "kernels/cuda_check.h"

namespace engine {

namespace {

struct Planned {
    bool prefill = false;
    std::vector<SeqId> sampled;
    std::vector<std::vector<int32_t>> drafts;
    PagedBatch batch;
};

Planned plan(Scheduler& scheduler, int max_blocks) {
    Step step;
    ranged("schedule", [&] { step = scheduler.schedule(); });
    if (step.seqs.empty())
        throw std::logic_error("serve: the scheduler returned an empty step with work left");
    Planned planned{step.prefill, step.seqs, step.drafts, {}};
    ranged("build_batch", [&] { planned.batch = build_paged_batch(step, scheduler, max_blocks); });
    return planned;
}

Planned plan_chunked(Scheduler& scheduler, int max_blocks, int token_budget) {
    ChunkedStep step;
    ranged("schedule", [&] { step = scheduler.schedule_chunked(token_budget); });
    if (step.empty())
        throw std::logic_error("serve: the scheduler returned an empty step with work left");
    Planned planned{step.decode.empty(), step.sampled, step.drafts, {}};
    ranged("build_batch",
           [&] { planned.batch = build_chunked_batch(step, scheduler, max_blocks); });
    return planned;
}

}  // namespace

std::vector<Request> serve_step(Scheduler& scheduler, PagedRunner& runner, Sampler& sampler,
                                DeviceBuffer& next, ServeStats* stats, DecodeGraphs* graphs,
                                std::vector<StepTrace>* trace, int token_budget) {
    const NvtxRange step_range("step");
    const Clock::time_point started = Clock::now();
    const int max_blocks = int(runner.max_blocks());
    const Planned step = token_budget > 0 ? plan_chunked(scheduler, max_blocks, token_budget)
                                          : plan(scheduler, max_blocks);
    const PagedBatch& batch = step.batch;
    if (stats) {
        ++stats->steps;
        stats->prefill_steps += step.prefill;
        const int free_blocks = scheduler.blocks().num_free();
        if (stats->min_free_blocks < 0 || free_blocks < stats->min_free_blocks)
            stats->min_free_blocks = free_blocks;
    }

    Tensor logits;
    ranged(batch.prefill ? "forward_prefill" : "forward_decode", [&] {
        logits = graphs && graphs->covers(batch) ? graphs->forward(batch) : runner.forward(batch);
    });
    const std::vector<SampleRow> rows = sample_rows(step.sampled, scheduler, step.drafts);
    const Tensor ids(next.data(), DType::I32, {batch.num_seqs});
    ranged("sample", [&] { sampler(ids, logits, rows, runner.stream()); });

    std::vector<int32_t> tokens(size_t(batch.num_seqs));
    ranged("read_tokens", [&] {
        CUDA_CHECK(cudaMemcpyAsync(tokens.data(), next.data(), tokens.size() * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, runner.stream()));
        CUDA_CHECK(cudaStreamSynchronize(runner.stream()));
    });
    std::vector<size_t> emitted_before;
    for (const SeqId id : step.sampled)
        emitted_before.push_back(scheduler.request(id).output.size());
    const DraftCounts drafts_before = scheduler.draft_counts();
    ranged("update", [&] { scheduler.update(tokens); });
    const int64_t drafted = scheduler.draft_counts().proposed - drafts_before.proposed;
    const int64_t accepted = scheduler.draft_counts().accepted - drafts_before.accepted;
    if (stats) {
        stats->drafted += drafted;
        stats->accepted += accepted;
    }
    std::vector<Request> finished = scheduler.take_finished();
    if (trace) {
        StepTrace& record = trace->emplace_back();
        record.start = started;
        record.end = Clock::now();
        record.prefill = step.prefill;
        for (size_t i = 0; i < step.sampled.size(); ++i) {
            const SeqId id = step.sampled[i];
            const auto done = std::find_if(finished.begin(), finished.end(),
                                           [id](const Request& r) { return r.id == id; });
            const Request& request = done != finished.end() ? *done : scheduler.request(id);
            for (size_t j = emitted_before[i]; j < request.output.size(); ++j) {
                record.seqs.push_back(id);
                record.tokens.push_back(request.output[j]);
            }
        }
        record.drafted = int(drafted);
        record.accepted = int(accepted);
        for (const Request& r : finished) record.finished.push_back(r.id);
        record.waiting = scheduler.num_waiting();
        record.running = scheduler.num_running();
        record.free_blocks = scheduler.blocks().num_free();
    }
    return finished;
}

std::vector<Request> serve(Scheduler& scheduler, PagedRunner& runner, Sampler& sampler,
                           DecodeGraphs* graphs, int token_budget) {
    if (sampler.max_rows() < runner.max_seqs())
        throw std::invalid_argument("serve: the sampler must take at least max_seqs rows");

    DeviceBuffer next(size_t(runner.max_seqs()) * sizeof(int32_t));
    std::vector<Request> done;
    while (scheduler.num_waiting() + scheduler.num_running() > 0) {
        std::vector<Request> finished =
            serve_step(scheduler, runner, sampler, next, nullptr, graphs, nullptr, token_budget);
        done.insert(done.end(), std::make_move_iterator(finished.begin()),
                    std::make_move_iterator(finished.end()));
    }
    return done;
}

}  // namespace engine
