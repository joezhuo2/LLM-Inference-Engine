#pragma once

#include <cstdint>

#include "engine/bench/workload.h"
#include "engine/runtime/model_runner.h"
#include "engine/runtime/paged_runner.h"
#include "engine/runtime/sampler.h"
#include "engine/runtime/serve.h"
#include "engine/sched/request.h"
#include "engine/sched/scheduler.h"

namespace engine {

void run_one_at_a_time(ModelRunner& runner, Workload& workload, int32_t eos_id,
                       Clock::time_point start);

ServeStats run_batched(Scheduler& scheduler, PagedRunner& runner, Sampler& sampler,
                       Workload& workload, Clock::time_point start);

}  // namespace engine
