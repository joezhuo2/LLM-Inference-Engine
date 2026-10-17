#include "engine/runtime/serve.h"

#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <utility>

#include "engine/runtime/device_buffer.h"
#include "engine/runtime/nvtx.h"
#include "kernels/cuda_check.h"

namespace engine {

std::vector<Request> serve_step(Scheduler& scheduler, PagedRunner& runner, Sampler& sampler,
                                DeviceBuffer& next, ServeStats* stats, DecodeGraphs* graphs) {
    const NvtxRange step_range("step");
    Step step;
    ranged("schedule", [&] { step = scheduler.schedule(); });
    if (step.seqs.empty())
        throw std::logic_error("serve: the scheduler returned an empty step with work left");
    if (stats) {
        ++stats->steps;
        stats->prefill_steps += step.prefill;
        const int free_blocks = scheduler.blocks().num_free();
        if (stats->min_free_blocks < 0 || free_blocks < stats->min_free_blocks)
            stats->min_free_blocks = free_blocks;
    }

    PagedBatch batch;
    ranged("build_batch",
           [&] { batch = build_paged_batch(step, scheduler, int(runner.max_blocks())); });
    Tensor logits;
    ranged(step.prefill ? "forward_prefill" : "forward_decode", [&] {
        logits = graphs && graphs->covers(batch) ? graphs->forward(batch) : runner.forward(batch);
    });
    const std::vector<SampleRow> rows = sample_rows(step, scheduler);
    const Tensor ids(next.data(), DType::I32, {batch.num_seqs});
    ranged("sample", [&] { sampler(ids, logits, rows, runner.stream()); });

    std::vector<int32_t> tokens(size_t(batch.num_seqs));
    ranged("read_tokens", [&] {
        CUDA_CHECK(cudaMemcpyAsync(tokens.data(), next.data(), tokens.size() * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, runner.stream()));
        CUDA_CHECK(cudaStreamSynchronize(runner.stream()));
    });
    ranged("update", [&] { scheduler.update(tokens); });
    return scheduler.take_finished();
}

std::vector<Request> serve(Scheduler& scheduler, PagedRunner& runner, Sampler& sampler,
                           DecodeGraphs* graphs) {
    if (sampler.max_rows() < runner.max_seqs())
        throw std::invalid_argument("serve: the sampler must take at least max_seqs rows");

    DeviceBuffer next(size_t(runner.max_seqs()) * sizeof(int32_t));
    std::vector<Request> done;
    while (scheduler.num_waiting() + scheduler.num_running() > 0) {
        std::vector<Request> finished =
            serve_step(scheduler, runner, sampler, next, nullptr, graphs);
        done.insert(done.end(), std::make_move_iterator(finished.begin()),
                    std::make_move_iterator(finished.end()));
    }
    return done;
}

}  // namespace engine
