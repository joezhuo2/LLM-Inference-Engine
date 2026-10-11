#include "engine/bench/workload.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine {

namespace {

class SplitMix64 {
public:
    explicit SplitMix64(uint64_t seed) : state_(seed) {}

    uint64_t next() {
        uint64_t z = state_ += 0x9e3779b97f4a7c15ull;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }

    uint64_t below(uint64_t n) {
        const uint64_t limit =
            std::numeric_limits<uint64_t>::max() - std::numeric_limits<uint64_t>::max() % n;
        uint64_t x = next();
        while (x >= limit) x = next();
        return x % n;
    }

    int between(LengthRange r) { return r.min + int(below(uint64_t(r.max - r.min) + 1)); }

    double unit() { return double(next() >> 11) * 0x1.0p-53; }

private:
    uint64_t state_;
};

int parse_positive(std::string_view text, std::string_view whole) {
    int value = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || end != text.data() + text.size() || value < 1)
        throw std::invalid_argument("length range '" + std::string(whole) +
                                    "': expected N or MIN:MAX with positive integers");
    return value;
}

void check(const LengthRange& r, const char* name) {
    if (r.min < 1 || r.max < r.min)
        throw std::invalid_argument(std::string("workload: ") + name +
                                    " must satisfy 1 <= min <= max");
}

}  // namespace

LengthRange parse_length_range(std::string_view text) {
    const auto colon = text.find(':');
    if (colon == std::string_view::npos) {
        const int n = parse_positive(text, text);
        return {n, n};
    }
    const LengthRange r{parse_positive(text.substr(0, colon), text),
                        parse_positive(text.substr(colon + 1), text)};
    if (r.max < r.min)
        throw std::invalid_argument("length range '" + std::string(text) +
                                    "': MIN is larger than MAX");
    return r;
}

Workload make_workload(const WorkloadConfig& config) {
    if (config.num_requests < 1)
        throw std::invalid_argument("workload: num_requests must be positive");
    check(config.prompt_len, "prompt_len");
    check(config.output_len, "output_len");
    if (config.vocab < 3) throw std::invalid_argument("workload: vocab must be at least 3");
    if (config.bos_id < 0 || config.bos_id >= config.vocab || config.eos_id < 0 ||
        config.eos_id >= config.vocab || config.bos_id == config.eos_id)
        throw std::invalid_argument(
            "workload: bos_id and eos_id must be distinct ids in the vocab");
    if (!(config.request_rate >= 0.0) || std::isinf(config.request_rate))
        throw std::invalid_argument("workload: request_rate must be finite and not negative");
    validate(config.params);

    SplitMix64 rng(config.seed);
    Workload w;
    w.requests.reserve(size_t(config.num_requests));
    w.arrivals.reserve(size_t(config.num_requests));
    double t = 0.0;
    for (int i = 0; i < config.num_requests; ++i) {
        Request r{.id = i, .max_new_tokens = rng.between(config.output_len)};
        r.params = config.params;
        r.params.seed = config.params.seed + uint64_t(i);
        r.prompt.resize(size_t(rng.between(config.prompt_len)));
        r.prompt[0] = config.bos_id;
        for (size_t j = 1; j < r.prompt.size(); ++j) {
            auto token = int32_t(rng.below(uint64_t(config.vocab - 2)));
            for (const int32_t special :
                 {std::min(config.bos_id, config.eos_id), std::max(config.bos_id, config.eos_id)})
                if (token >= special) ++token;
            r.prompt[j] = token;
        }
        if (i > 0 && config.request_rate > 0.0) t -= std::log1p(-rng.unit()) / config.request_rate;
        w.arrivals.push_back(t);
        w.requests.push_back(std::move(r));
    }
    return w;
}

}  // namespace engine
