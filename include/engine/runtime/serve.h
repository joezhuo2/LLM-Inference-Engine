#pragma once

#include <vector>

#include "engine/runtime/paged_runner.h"
#include "engine/runtime/sampler.h"
#include "engine/sched/scheduler.h"

namespace engine {

std::vector<Request> serve(Scheduler& scheduler, PagedRunner& runner, Sampler& sampler);

}  // namespace engine
