#pragma once

#include <vector>

#include "engine/bench/trace.h"
#include "engine/sched/request.h"
#include "engine/sched/scheduler.h"

namespace engine::server {

struct StepResult {
    StepTrace trace{};
    std::vector<Request> finished{};
};

class Backend {
public:
    virtual ~Backend() = default;

    virtual Scheduler& scheduler() = 0;
    virtual int num_blocks() const = 0;
    virtual StepResult step() = 0;
};

}  // namespace engine::server
