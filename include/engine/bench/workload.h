#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "engine/sampling/params.h"
#include "engine/sched/request.h"

namespace engine {

struct LengthRange {
    int min = 1;
    int max = 1;
};

LengthRange parse_length_range(std::string_view text);

struct WorkloadConfig {
    int num_requests = 1;
    LengthRange prompt_len{128, 128};
    LengthRange output_len{256, 256};
    int vocab = 32000;
    int32_t bos_id = 1;
    int32_t eos_id = 2;
    double request_rate = 0.0;
    uint64_t seed = 0;
    SamplingParams params{};
    std::vector<std::vector<int32_t>> prompts{};
};

struct Workload {
    std::vector<Request> requests;
    std::vector<double> arrivals;
};

Workload make_workload(const WorkloadConfig& config);

}  // namespace engine
