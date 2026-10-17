#pragma once

#include <vector>

#include "engine/runtime/decode_graphs.h"
#include "engine/runtime/device_buffer.h"
#include "engine/runtime/paged_runner.h"
#include "engine/runtime/sampler.h"
#include "engine/sched/scheduler.h"

namespace engine {

struct ServeStats {
    int steps = 0;
    int prefill_steps = 0;
    int min_free_blocks = -1;
};

std::vector<Request> serve_step(Scheduler& scheduler, PagedRunner& runner, Sampler& sampler,
                                DeviceBuffer& next, ServeStats* stats = nullptr,
                                DecodeGraphs* graphs = nullptr);

std::vector<Request> serve(Scheduler& scheduler, PagedRunner& runner, Sampler& sampler,
                           DecodeGraphs* graphs = nullptr);

}  // namespace engine
