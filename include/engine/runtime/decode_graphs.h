#pragma once

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "engine/core/tensor.h"
#include "engine/runtime/paged_runner.h"
#include "engine/sched/paged_batch.h"

struct CUgraphExec_st;

namespace engine {

class DecodeGraphs {
public:
    DecodeGraphs(PagedRunner& runner, std::vector<int64_t> buckets);
    ~DecodeGraphs();

    DecodeGraphs(const DecodeGraphs&) = delete;
    DecodeGraphs& operator=(const DecodeGraphs&) = delete;

    bool covers(const PagedBatch& batch) const;
    int64_t bucket_for(int64_t seqs) const;
    int64_t num_captured() const;

    Tensor forward(const PagedBatch& batch);

private:
    PagedRunner* runner_;
    std::vector<int64_t> buckets_;
    std::map<std::pair<int64_t, int64_t>, CUgraphExec_st*> graphs_;
};

}  // namespace engine
