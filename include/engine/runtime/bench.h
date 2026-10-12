#pragma once

#include <cstdint>

#include "engine/bench/workload.h"
#include "engine/runtime/model_runner.h"
#include "engine/sched/request.h"

namespace engine {

void run_one_at_a_time(ModelRunner& runner, Workload& workload, int32_t eos_id,
                       Clock::time_point start);

}  // namespace engine
