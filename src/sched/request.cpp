#include "engine/sched/request.h"

#include <stdexcept>
#include <string>

namespace engine {

void validate(const Request& request, int max_context) {
    const std::string name = "request " + std::to_string(request.id);
    if (request.prompt.empty()) throw std::invalid_argument(name + ": the prompt is empty");
    if (request.max_new_tokens < 1)
        throw std::invalid_argument(name + ": max_new_tokens must be positive");
    if (int64_t(request.prompt.size()) + request.max_new_tokens - 1 > max_context)
        throw std::invalid_argument(name + ": the prompt plus max_new_tokens does not fit in " +
                                    std::to_string(max_context) + " tokens of context");
    if (!request.output.empty()) throw std::invalid_argument(name + ": the output is not empty");
    validate(request.params);
}

}  // namespace engine
