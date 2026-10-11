#pragma once

#include <chrono>
#include <cstdint>
#include <vector>

#include "engine/kv/block_manager.h"
#include "engine/sampling/params.h"

namespace engine {

using Clock = std::chrono::steady_clock;

enum class RequestState { Waiting, Running, Finished };

struct Request {
    SeqId id = 0;
    std::vector<int32_t> prompt{};
    int max_new_tokens = 1;
    SamplingParams params{};
    std::vector<int32_t> output{};
    RequestState state = RequestState::Waiting;
    int preemptions = 0;
    Clock::time_point arrived{}, first_token{}, finished{};

    int64_t length() const { return int64_t(prompt.size() + output.size()); }
    int32_t token(int64_t pos) const {
        const auto n = int64_t(prompt.size());
        return pos < n ? prompt[pos] : output[pos - n];
    }
};

void validate(const Request& request, int max_context);

}  // namespace engine
