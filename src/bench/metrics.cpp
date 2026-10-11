#include "engine/bench/metrics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace engine {

namespace {

double seconds(Clock::duration d) {
    return std::chrono::duration<double>(d).count();
}

Stat stat(std::span<const RequestMetrics> metrics, double RequestMetrics::*field) {
    std::vector<double> values;
    for (const auto& m : metrics)
        if (!std::isnan(m.*field)) values.push_back(m.*field);
    Stat s;
    s.mean = std::numeric_limits<double>::quiet_NaN();
    if (!values.empty()) {
        double sum = 0.0;
        for (const double v : values) sum += v;
        s.mean = sum / double(values.size());
    }
    s.p50 = percentile(values, 50.0);
    s.p90 = percentile(values, 90.0);
    s.p99 = percentile(values, 99.0);
    return s;
}

}  // namespace

RequestMetrics request_metrics(const Request& r, Clock::time_point start) {
    const std::string name = "request " + std::to_string(r.id);
    if (r.state != RequestState::Finished || r.output.empty())
        throw std::invalid_argument(name + ": metrics need a finished request with output");
    if (r.arrived < start || r.first_token < r.arrived || r.finished < r.first_token)
        throw std::invalid_argument(name +
                                    ": timestamps must satisfy start <= arrived <= first_token "
                                    "<= finished");
    const auto n = int64_t(r.output.size());
    return {
        .id = r.id,
        .prompt_tokens = int64_t(r.prompt.size()),
        .output_tokens = n,
        .preemptions = r.preemptions,
        .arrival_s = seconds(r.arrived - start),
        .ttft_s = seconds(r.first_token - r.arrived),
        .tpot_s = n > 1 ? seconds(r.finished - r.first_token) / double(n - 1)
                        : std::numeric_limits<double>::quiet_NaN(),
        .latency_s = seconds(r.finished - r.arrived),
    };
}

double percentile(std::vector<double> values, double q) {
    if (!(q >= 0.0 && q <= 100.0)) throw std::invalid_argument("percentile: q must be in [0, 100]");
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const double rank = q / 100.0 * double(values.size() - 1);
    const auto lo = size_t(std::floor(rank));
    const size_t hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (rank - double(lo)) * (values[hi] - values[lo]);
}

Summary summarize(std::span<const RequestMetrics> metrics, double wall_s) {
    if (metrics.empty()) throw std::invalid_argument("summarize: no requests");
    if (!(wall_s > 0.0) || std::isinf(wall_s))
        throw std::invalid_argument("summarize: wall_s must be positive and finite");
    Summary s;
    s.requests = int64_t(metrics.size());
    for (const auto& m : metrics) {
        s.prompt_tokens += m.prompt_tokens;
        s.output_tokens += m.output_tokens;
    }
    s.wall_s = wall_s;
    s.request_throughput = double(s.requests) / wall_s;
    s.output_throughput = double(s.output_tokens) / wall_s;
    s.ttft_s = stat(metrics, &RequestMetrics::ttft_s);
    s.tpot_s = stat(metrics, &RequestMetrics::tpot_s);
    s.latency_s = stat(metrics, &RequestMetrics::latency_s);
    return s;
}

}  // namespace engine
