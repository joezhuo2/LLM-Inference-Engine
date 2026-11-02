#include "engine/bench/trace.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine {

namespace {

double seconds_since(Clock::time_point start, Clock::time_point t) {
    return std::chrono::duration<double>(t - start).count();
}

}  // namespace

nlohmann::ordered_json step_json(const StepTrace& step, size_t index, Clock::time_point start,
                                 std::span<const std::string> texts) {
    if (step.tokens.size() != step.seqs.size() || texts.size() != step.seqs.size())
        throw std::invalid_argument("step_json: a step needs one token and one text per sequence");
    nlohmann::ordered_json tokens = nlohmann::ordered_json::array();
    for (size_t j = 0; j < step.seqs.size(); ++j)
        tokens.push_back({{"id", step.tokens[j]}, {"text", texts[j]}});
    return {{"index", index},
            {"start_s", seconds_since(start, step.start)},
            {"end_s", seconds_since(start, step.end)},
            {"kind", step.prefill ? "prefill" : "decode"},
            {"seqs", step.seqs},
            {"tokens", std::move(tokens)},
            {"finished", step.finished},
            {"waiting", step.waiting},
            {"running", step.running},
            {"free_blocks", step.free_blocks}};
}

nlohmann::ordered_json trace_json(nlohmann::ordered_json meta, std::span<const Request> requests,
                                  std::span<const StepTrace> steps, Clock::time_point start,
                                  const TokenText& text) {
    const auto seconds = [&](Clock::time_point t) { return seconds_since(start, t); };

    std::vector<const Request*> by_id;
    for (const Request& r : requests) by_id.push_back(&r);
    std::sort(by_id.begin(), by_id.end(),
              [](const Request* a, const Request* b) { return a->id < b->id; });
    nlohmann::ordered_json request_list = nlohmann::ordered_json::array();
    for (const Request* r : by_id)
        request_list.push_back({{"id", r->id},
                                {"arrival_s", seconds(r->arrived)},
                                {"prompt_tokens", r->prompt.size()},
                                {"output_tokens", r->output.size()},
                                {"preemptions", r->preemptions},
                                {"first_token_s", seconds(r->first_token)},
                                {"finished_s", seconds(r->finished)}});

    nlohmann::ordered_json step_list = nlohmann::ordered_json::array();
    for (size_t i = 0; i < steps.size(); ++i) {
        const StepTrace& s = steps[i];
        if (s.tokens.size() != s.seqs.size())
            throw std::invalid_argument("trace_json: a step needs one token per sequence");
        std::vector<std::string> texts;
        for (size_t j = 0; j < s.seqs.size(); ++j) texts.push_back(text(s.seqs[j], s.tokens[j]));
        step_list.push_back(step_json(s, i, start, texts));
    }

    return {{"format", "llmie-trace"},
            {"version", 1},
            {"meta", std::move(meta)},
            {"requests", std::move(request_list)},
            {"steps", std::move(step_list)}};
}

}  // namespace engine
