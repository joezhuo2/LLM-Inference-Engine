#include "engine/sched/graph_buckets.h"

#include <stdexcept>

namespace engine {

/*
TODO(JOE): decode_graph_buckets

Purpose. DecodeGraphs pads every decode batch up to the smallest captured bucket that holds it, and today the buckets are the powers of two below --max-batch plus --max-batch itself (graph_buckets in src/app/bench_command.cpp and again in src/server/gpu_backend.cpp). Up to batch 16 that wastes little, but #121 sweeps 32, 64, 128 and 256, and there 17 sequences under --max-batch 32 run 32 rows, 33 under 64 run 64, and so on: just under half the rows can be padding, and with the GEMMs bandwidth bound until roughly batch 100 (16 FLOP/byte against a 107 ridge) the padded rows still cost attention, the LM head and sampling. This function makes the bucket grid finer on request, so the sweep can measure what padding costs at each size and whether a finer grid pays for the extra graphs. steps_per_octave = 1 is exactly today's grid.

Interface. std::vector<int64_t> decode_graph_buckets(int64_t max_batch, int64_t steps_per_octave), declared in include/engine/sched/graph_buckets.h, part of engine_core so it runs in the cpu preset. It returns the list DecodeGraphs takes as its buckets argument: strictly increasing, starting at 1 and ending at max_batch.

The grid. Split the range into octaves [2^k, 2^(k+1)). Within octave k the step is max(1, 2^k / steps_per_octave), and the octave contributes 2^k, 2^k + step, 2^k + 2 * step, and so on while the value stays below 2^(k+1). Keep every value below max_batch, then append max_batch. So steps_per_octave = 2 gives 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, ... and steps_per_octave = 4 gives 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, ... An octave smaller than steps_per_octave just takes every integer in it. max_batch itself appears once, whether or not it falls on the grid.

What must hold. Every power of two up to max_batch is a bucket, and a finer grid contains every bucket of a coarser one (steps_per_octave s and 2s). Two neighbouring buckets b and b' satisfy b' - b <= max(1, b / steps_per_octave), so a batch of n sequences pads by fewer than n / steps_per_octave rows: under 50% of its rows padding at 1, under a third at 2, under a fifth at 4. With steps_per_octave = 1 the result equals the current graph_buckets for every max_batch, so swapping it in changes nothing until someone asks for more.

Errors. Throw std::invalid_argument when max_batch is below 1 or steps_per_octave is not a power of two (1, 2, 4, 8, ...). No upper limit on either here; DecodeGraphs already rejects buckets above the runner's max_seqs.

Edge cases. max_batch = 1 gives {1} for any steps_per_octave. max_batch on the grid is not repeated (steps_per_octave = 2, max_batch = 12 ends 8, 12). max_batch off the grid ends the list after the last grid value below it (steps_per_octave = 2, max_batch = 13 ends 8, 12, 13). A steps_per_octave larger than max_batch gives every integer from 1 to max_batch.

Cost to keep in mind. DecodeGraphs captures one graph per bucket and attention split count, so the bucket count multiplies capture time at start-up and graph memory: to 256 the grid has 9 buckets at steps_per_octave 1, 16 at 2 and 28 at 4. Check num_captured() and start-up time when you pick a default.

What is not here. Wiring comes after this passes: replace both graph_buckets copies with a call to this, and add a --graph-steps option (default 1) to engine bench and engine serve. The Part A half of #121 (the 32 to 256 sweep with and without graphs, the per-op profile at each size, the padding cost of 17 sequences under --max-batch 32) then runs at steps_per_octave 1 and 2 or 4 and decides whether the default changes.

Tests that prove it. tests/todo/unit/test_graph_buckets.cpp (target todo_unit_tests, label todo_joe). Run them with ctest --preset cpu -L todo_joe. When they pass, move the file into tests/unit/, add it to unit_tests and delete todo_unit_tests.
*/
std::vector<int64_t> decode_graph_buckets(int64_t, int64_t) {
    throw std::logic_error("unimplemented: decode_graph_buckets");
}

}  // namespace engine
