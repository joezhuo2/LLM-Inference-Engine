#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "engine/tokenizer/tokenizer.h"

namespace engine {

std::vector<std::vector<int32_t>> load_chat_prompts(const std::filesystem::path& file,
                                                    const Tokenizer& tokenizer);

}  // namespace engine
