#include "engine/runtime/bench.h"

#include <chrono>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "engine/runtime/device_buffer.h"
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

ServeStats run_batched(Scheduler& scheduler, PagedRunner& runner, Sampler& sampler,
                       Workload& workload, Clock::time_point start, DecodeGraphs* graphs,
                       std::vector<StepTrace>* trace) {
    auto& requests = workload.requests;
    if (workload.arrivals.size() != requests.size())
        throw std::invalid_argument("run_batched: one arrival time per request");
    if (sampler.max_rows() < runner.max_seqs())
        throw std::invalid_argument("run_batched: the sampler must take at least max_seqs rows");
    std::unordered_map<SeqId, size_t> index;
    for (size_t i = 0; i < requests.size(); ++i) {
        if (!index.emplace(requests[i].id, i).second)
            throw std::invalid_argument("run_batched: duplicate request id");
    }

    const auto arrival = [&](size_t i) {
        return start + std::chrono::duration_cast<Clock::duration>(
                           std::chrono::duration<double>(workload.arrivals[i]));
    };
    DeviceBuffer next(size_t(runner.max_seqs()) * sizeof(int32_t));
    ServeStats stats;
    size_t submitted = 0, finished = 0;
    while (finished < requests.size()) {
        while (submitted < requests.size() && arrival(submitted) <= Clock::now())
            scheduler.submit(requests[submitted++]);
        if (scheduler.num_waiting() + scheduler.num_running() == 0) {
            std::this_thread::sleep_until(arrival(submitted));
            continue;
        }
        for (Request& done : serve_step(scheduler, runner, sampler, next, &stats, graphs, trace)) {
            const size_t i = index.at(done.id);
            done.arrived = arrival(i);
            requests[i] = std::move(done);
            ++finished;
        }
    }
    return stats;
}

}  // namespace engine
