#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "engine/sched/request.h"

namespace engine {

struct RequestMetrics {
    SeqId id = 0;
    int64_t prompt_tokens = 0;
    int64_t output_tokens = 0;
    int preemptions = 0;
    double arrival_s = 0.0;
    double ttft_s = 0.0;
    double tpot_s = 0.0;
    double latency_s = 0.0;
};

RequestMetrics request_metrics(const Request& request, Clock::time_point start);

double percentile(std::vector<double> values, double q);

struct Stat {
    double mean = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p99 = 0.0;
};

struct Summary {
    int64_t requests = 0;
    int64_t prompt_tokens = 0;
    int64_t output_tokens = 0;
    double wall_s = 0.0;
    double request_throughput = 0.0;
    double output_throughput = 0.0;
    Stat ttft_s{}, tpot_s{}, latency_s{};
};

Summary summarize(std::span<const RequestMetrics> metrics, double wall_s);

}  // namespace engine
