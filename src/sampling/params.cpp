#include "engine/sampling/params.h"

#include <cmath>
#include <stdexcept>

namespace engine {

void validate(const SamplingParams& params) {
    if (!(params.temperature >= 0) || !std::isfinite(params.temperature))
        throw std::invalid_argument("sampling: temperature must be zero or positive and finite");
    if (params.top_k < 0)
        throw std::invalid_argument("sampling: top_k must be zero (off) or positive");
    if (!(params.top_p > 0 && params.top_p <= 1))
        throw std::invalid_argument("sampling: top_p must be in (0, 1]");
}

}  // namespace engine
