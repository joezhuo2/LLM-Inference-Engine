#include "engine/runtime/decode_graphs.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "kernels/cuda_check.h"

namespace engine {

namespace {
void validate_buckets(const std::vector<int64_t>& buckets, const PagedRunner& runner) {
    if (buckets.empty()) {
        throw std::invalid_argument("[DecodeGraphs] bucket list cannot be empty");
    }

    for (size_t i = 0; i < buckets.size(); ++i) {
        int64_t b = buckets[i];

        if (b <= 0) {
            throw std::invalid_argument("[DecodeGraphs] bucket must be positive");
        }

        if (i > 0 && b <= buckets[i - 1]) {
            throw std::invalid_argument("[DecodeGraphs] buckets must be strictly increasing");
        }

        if (b > runner.max_seqs() || b > runner.max_tokens()) {
            throw std::invalid_argument(
                "[DecodeGraphs] bucket exceeds runner max_seqs or max_tokens");
        }
    }
}

cudaGraphExec_t capture(PagedRunner& runner, const PagedBatch& padded) {
    const int64_t b = padded.num_seqs;
    runner.stage(padded);
    runner.run_staged(b, b);

    CUDA_CHECK(cudaStreamBeginCapture(runner.stream(), cudaStreamCaptureModeGlobal));

    runner.run_staged(b, b);

    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(runner.stream(), &graph));

    cudaGraphExec_t exec = nullptr;
    const cudaError_t status = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    CUDA_CHECK(status);
    return exec;
}
}  // namespace

DecodeGraphs::DecodeGraphs(PagedRunner& runner, std::vector<int64_t> buckets)
    : runner_(&runner), buckets_(std::move(buckets)) {
    validate_buckets(buckets_, *runner_);

    PagedBatch dummy;
    dummy.prefill = false;
    dummy.num_seqs = 1;
    dummy.max_blocks = runner_->max_blocks();
    dummy.token_ids = {0};
    dummy.positions = {0};
    dummy.slot_mapping = {-1};
    dummy.context_lens = {0};
    dummy.logit_rows = {0};
    dummy.block_tables.resize(size_t(runner_->max_blocks()), 0);

    const int64_t heads = runner_->config().heads;
    for (int64_t b : buckets_) {
        for (int64_t context = 0; context <= runner_->max_context(); ++context) {
            const int64_t splits = attention_splits(b, heads, context);
            if (graphs_.contains({b, splits})) {
                continue;
            }

            dummy.context_lens = {int32_t(context)};
            graphs_[{b, splits}] = capture(*runner_, pad_decode_batch(dummy, b));
        }
    }
}

DecodeGraphs::~DecodeGraphs() {
    if (runner_) {
        cudaStreamSynchronize(runner_->stream());
    }

    for (auto& [key, exec] : graphs_) {
        cudaGraphExecDestroy(exec);
    }
}

bool DecodeGraphs::covers(const PagedBatch& batch) const {
    if (batch.prefill) {
        return false;
    }

    if (batch.num_seqs < 1 || batch.num_tokens() != batch.num_seqs) {
        return false;
    }

    if (batch.max_blocks != runner_->max_blocks()) {
        return false;
    }

    if (batch.longest_context() > runner_->max_context()) {
        return false;
    }

    return bucket_for(batch.num_seqs) != 0;
}

int64_t DecodeGraphs::bucket_for(int64_t seqs) const {
    if (seqs <= 0) {
        return 0;
    }

    for (int64_t b : buckets_) {
        if (seqs <= b) {
            return b;
        }
    }

    return 0;
}

int64_t DecodeGraphs::num_captured() const {
    return int64_t(graphs_.size());
}

Tensor DecodeGraphs::forward(const PagedBatch& batch) {
    if (!covers(batch)) {
        throw std::invalid_argument("[DecodeGraphs] batch is not covered");
    }

    int64_t b = bucket_for(batch.num_seqs);
    PagedBatch padded = pad_decode_batch(batch, b);

    const int64_t splits = attention_splits(b, runner_->config().heads, padded.longest_context());

    runner_->stage(padded);

    CUDA_CHECK(cudaGraphLaunch(graphs_.at({b, splits}), runner_->stream()));

    return runner_->logits(batch.num_seqs);
}

}  // namespace engine
