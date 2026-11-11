#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace engine {

std::vector<int32_t> propose_prompt_lookup(std::span<const int32_t> context, int max_ngram,
                                           int max_draft);
int count_accepted(std::span<const int32_t> draft, std::span<const int32_t> sampled);

}  // namespace engine
