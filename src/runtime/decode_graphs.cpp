#include "engine/runtime/decode_graphs.h"

#include <cstdio>
#include <cstdlib>
#include <utility>

namespace engine {

/*
TODO(JOE): DecodeGraphs

Purpose. A decode step of TinyLlama is about 300 small kernel launches (22 layers times RMSNorm, QKV GEMM, RoPE, write_kv, attention, output GEMM, add, RMSNorm, gate/up GEMM, SwiGLU, down GEMM, add) and each launch costs the CPU a few microseconds, which at batch 1 is as much as the kernels themselves take: the GPU idles between kernels waiting for the host. A CUDA Graph records the whole sequence once and replays it with a single cudaGraphLaunch, so the launches are submitted together and the gaps in the nsys timeline disappear. That is half of the M4 exit criterion ("nsys timeline shows no launch gaps"; the other half is 60% of measured bandwidth at batch 1). DecodeGraphs owns one captured graph per batch-size bucket (1, 2, 4, 8, 16) of PagedRunner's decode step and replays the one that fits a decode batch. Prefill is not graphed: its shape changes every step.

Interface. DecodeGraphs(runner, buckets) takes the runner by reference (it must outlive this object and must not be used from another thread while a graph is captured or replayed) and the bucket sizes, strictly increasing positive integers each at most runner.max_seqs() and runner.max_tokens(). covers(batch) is true exactly when the batch is a decode batch (prefill false, num_tokens() = num_seqs >= 1), num_seqs is at most the largest bucket and batch.max_blocks equals runner.max_blocks(). bucket_for(seqs) is the smallest bucket that is at least seqs, or 0 when there is none. num_captured() is the number of graphs captured so far. forward(batch) pads the batch to its bucket with pad_decode_batch (include/engine/sched/paged_batch.h: padded rows write nothing to the KV cache and attend to nothing), stages it with runner.stage, launches the bucket's graph on runner.stream() and returns the logits as a BF16 tensor [batch.num_seqs, vocab], a view of the first rows of the runner's logits buffer, valid until the runner's next forward. Like PagedRunner::forward it is asynchronous on the runner's stream: the caller synchronizes (the sampler's copy back does).

How a graph is made. The kernels of PagedRunner::run_staged only read device buffers that are allocated once (the packed int inputs, the activations, the KV cache, the weights) and the host never touches the data between kernels, which is what makes a capture replayable: every replay reads the new token ids, positions, slots, context lengths and block tables that stage() has just copied to the same device addresses, and runs the same kernels on them. Per bucket: run_staged(bucket, bucket) once eagerly on an all-padding batch first (cuBLAS picks its algorithm and loads its kernels on the first call, and neither can be captured), then cudaStreamBeginCapture on runner.stream(), run_staged(bucket, bucket), cudaStreamEndCapture, cudaGraphInstantiate, keeping the executable graph. Capture only works on a stream that is not the legacy default stream; nothing in the captured region may allocate, synchronize or copy from pageable host memory (the runner was changed so it does not), and PagedRunner::stage must stay outside the capture (the H2D copy of the batch is not part of the graph). Choose the capture mode with the cuBLAS and multi-threading hazards in mind and be ready to explain the choice. The destructor destroys every executable graph and graph; it must not throw and must not run while a replay is pending, so it synchronizes the stream first.

When graphs are captured. In the constructor, all of them, so the first real request does not pay for capture and a capture failure surfaces at startup. Capturing uses an all-padding batch, so the KV cache is left exactly as it was found (padded rows have slot -1 and context length 0): the tests snapshot the whole cache around the constructor.

Numerics. Replaying a graph runs the same kernels with the same arguments as runner.run_staged(bucket, bucket) eagerly on the same padded batch, so its logits are bit-identical to that eager run. They are not guaranteed bit-identical to runner.forward on the unpadded batch: the GEMMs run with bucket rows instead of num_seqs rows and cuBLAS may then choose another algorithm, which changes BF16 rounding, so the tests compare the unpadded eager run with the same tolerance as the paged runner tests (relative L2 at most 0.05) and the padded eager run exactly. Decoding a sequence through a graph over many steps must produce exactly the tokens the padded eager run produces, whatever the batch composition does between steps (a batch shrinking across buckets, sequences finishing, the cache blocks being reused).

Errors and preconditions. The constructor throws std::invalid_argument with a message naming DecodeGraphs for an empty bucket list, a bucket that is not positive, buckets not strictly increasing, and a bucket above runner.max_seqs() or runner.max_tokens(). forward throws std::invalid_argument when covers(batch) is false. CUDA errors go through CUDA_CHECK as everywhere else. Mixing graph replays and eager runner.forward calls on the same runner (prefill steps are eager, decode steps are replays) must work, since they share the runner's buffers and are ordered by the one stream.

Edge cases and boundaries. A bucket list with a single bucket (only batch 1, or only 16) works; a batch of exactly a bucket size needs no padding rows; a batch of one more than a bucket uses the next bucket; the same graph is replayed thousands of times without growing memory; forward may be called with batches whose sequences, positions and block tables are different every time (nothing about the first batch may be baked into the graph except shapes).

What to compare against. runner.run_staged(bucket, bucket) after runner.stage(pad_decode_batch(batch, bucket)) for exact equality, runner.forward(batch) for tolerance, and a full multi-step decode against the same loop run eagerly at padded size.

Tests that prove it. tests/todo/gpu/test_decode_graphs.cpp: bit identity with the padded eager step for batches of 1, 2, 3, 5, 8, 13 and 16, closeness to the unpadded eager step, a 16-sequence greedy decode in which sequences finish at different steps and the batch crosses every bucket, a cache snapshot before and after construction, 200 replays of one graph with changing inputs, interleaving eager prefill with replays, covers and bucket_for, single-bucket lists and every rejected argument. They run in the todo_gpu_tests target with the todo_joe label (ctest --preset debug -L todo_joe). When they all pass, move the file to tests/gpu/, add it to gpu_tests and delete todo_gpu_tests if it is empty; also run compute-sanitizer --tool memcheck on them, and record an nsys timeline of a 64-token decode at batch 1 with and without graphs (the launch gaps and the tokens per second) in docs. Wiring the graphs into serve() and the --graphs flag of engine bench is branch 23, once these pass.
*/
DecodeGraphs::DecodeGraphs(PagedRunner& runner, std::vector<int64_t> buckets)
    : runner_(&runner), buckets_(std::move(buckets)) {
    std::fprintf(stderr, "unimplemented: DecodeGraphs\n");
    std::abort();
}

DecodeGraphs::~DecodeGraphs() = default;

bool DecodeGraphs::covers(const PagedBatch&) const {
    std::fprintf(stderr, "unimplemented: DecodeGraphs::covers\n");
    std::abort();
}

int64_t DecodeGraphs::bucket_for(int64_t) const {
    std::fprintf(stderr, "unimplemented: DecodeGraphs::bucket_for\n");
    std::abort();
}

int64_t DecodeGraphs::num_captured() const {
    std::fprintf(stderr, "unimplemented: DecodeGraphs::num_captured\n");
    std::abort();
}

Tensor DecodeGraphs::forward(const PagedBatch&) {
    std::fprintf(stderr, "unimplemented: DecodeGraphs::forward\n");
    std::abort();
}

}  // namespace engine
