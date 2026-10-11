#include "engine/runtime/bench.h"

#include <chrono>
#include <stdexcept>
#include <thread>

#include "engine/runtime/generate.h"

namespace engine {

void run_one_at_a_time(ModelRunner& runner, Workload& workload, int32_t eos_id,
                       Clock::time_point start) {
    auto& requests = workload.requests;
    if (workload.arrivals.size() != requests.size())
        throw std::invalid_argument("run_one_at_a_time: one arrival time per request");
    for (const auto& r : requests) validate(r, int(runner.max_context()));

    for (size_t i = 0; i < requests.size(); ++i) {
        Request& r = requests[i];
        r.arrived = start + std::chrono::duration_cast<Clock::duration>(
                                std::chrono::duration<double>(workload.arrivals[i]));
        std::this_thread::sleep_until(r.arrived);
        r.state = RequestState::Running;
        r.output = generate(runner, r.prompt, r.max_new_tokens, eos_id, r.params, [&](int32_t) {
            r.finished = Clock::now();
            if (r.first_token == Clock::time_point{}) r.first_token = r.finished;
        });
        r.state = RequestState::Finished;
    }
}

}  // namespace engine
