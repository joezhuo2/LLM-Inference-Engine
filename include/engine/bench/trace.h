#pragma once

#include <cstdint>
#include <functional>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <vector>

#include "engine/sched/request.h"

namespace engine {

struct StepTrace {
    Clock::time_point start{}, end{};
    bool prefill = false;
    std::vector<SeqId> seqs{};
    std::vector<int32_t> tokens{};
    std::vector<SeqId> finished{};
    int waiting = 0;
    int running = 0;
    int free_blocks = 0;
};

using TokenText = std::function<std::string(SeqId seq, int32_t token)>;

nlohmann::ordered_json step_json(const StepTrace& step, size_t index, Clock::time_point start,
                                 std::span<const std::string> texts);

nlohmann::ordered_json trace_json(nlohmann::ordered_json meta, std::span<const Request> requests,
                                  std::span<const StepTrace> steps, Clock::time_point start,
                                  const TokenText& text);

}  // namespace engine
