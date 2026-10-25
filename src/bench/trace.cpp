#include "engine/bench/trace.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine {

nlohmann::ordered_json trace_json(nlohmann::ordered_json meta, std::span<const Request> requests,
                                  std::span<const StepTrace> steps, Clock::time_point start,
                                  const TokenText& text) {
    const auto seconds = [&](Clock::time_point t) {
        return std::chrono::duration<double>(t - start).count();
    };

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
        nlohmann::ordered_json tokens = nlohmann::ordered_json::array();
        for (size_t j = 0; j < s.seqs.size(); ++j)
            tokens.push_back({{"id", s.tokens[j]}, {"text", text(s.seqs[j], s.tokens[j])}});
        step_list.push_back({{"index", i},
                             {"start_s", seconds(s.start)},
                             {"end_s", seconds(s.end)},
                             {"kind", s.prefill ? "prefill" : "decode"},
                             {"seqs", s.seqs},
                             {"tokens", std::move(tokens)},
                             {"finished", s.finished},
                             {"waiting", s.waiting},
                             {"running", s.running},
                             {"free_blocks", s.free_blocks}});
    }

    return {{"format", "llmie-trace"},
            {"version", 1},
            {"meta", std::move(meta)},
            {"requests", std::move(request_list)},
            {"steps", std::move(step_list)}};
}

}  // namespace engine
