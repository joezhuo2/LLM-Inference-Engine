#include "engine/server/metrics_window.h"

#include <algorithm>
#include <vector>

#include "engine/bench/metrics.h"

namespace engine::server {

namespace {

std::optional<Percentiles> percentiles(const std::vector<double>& values) {
    if (values.empty()) return std::nullopt;
    return Percentiles{percentile(values, 50.0), percentile(values, 99.0)};
}

}  // namespace

void MetricsWindow::add_tokens(double time_s, int64_t count) {
    tokens_.push_back({time_s, count});
    trim(time_s);
}

void MetricsWindow::add_request(double time_s, double ttft_s, std::optional<double> tpot_s) {
    finished_.push_back({time_s, ttft_s, tpot_s});
    trim(time_s);
}

double MetricsWindow::output_per_s(double now_s) {
    trim(now_s);
    const double span = std::min(now_s, kThroughputWindowS);
    if (span <= 0.0) return 0.0;
    int64_t total = 0;
    for (const Tokens& t : tokens_) total += t.count;
    return double(total) / span;
}

std::optional<Percentiles> MetricsWindow::ttft_s(double now_s) {
    trim(now_s);
    std::vector<double> values;
    for (const Finished& f : finished_) values.push_back(f.ttft_s);
    return percentiles(values);
}

std::optional<Percentiles> MetricsWindow::tpot_s(double now_s) {
    trim(now_s);
    std::vector<double> values;
    for (const Finished& f : finished_)
        if (f.tpot_s) values.push_back(*f.tpot_s);
    return percentiles(values);
}

void MetricsWindow::trim(double now_s) {
    while (!tokens_.empty() && tokens_.front().time_s <= now_s - kThroughputWindowS)
        tokens_.pop_front();
    while (!finished_.empty() && finished_.front().time_s <= now_s - kLatencyWindowS)
        finished_.pop_front();
}

}  // namespace engine::server
