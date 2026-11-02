#pragma once

#include <cstdint>
#include <deque>
#include <optional>

namespace engine::server {

inline constexpr double kThroughputWindowS = 10.0;
inline constexpr double kLatencyWindowS = 60.0;

struct Percentiles {
    double p50 = 0.0;
    double p99 = 0.0;
};

class MetricsWindow {
public:
    void add_tokens(double time_s, int64_t count);
    void add_request(double time_s, double ttft_s, std::optional<double> tpot_s);

    double output_per_s(double now_s);
    std::optional<Percentiles> ttft_s(double now_s);
    std::optional<Percentiles> tpot_s(double now_s);

private:
    struct Tokens {
        double time_s;
        int64_t count;
    };
    struct Finished {
        double time_s;
        double ttft_s;
        std::optional<double> tpot_s;
    };

    std::deque<Tokens> tokens_;
    std::deque<Finished> finished_;

    void trim(double now_s);
};

}  // namespace engine::server
