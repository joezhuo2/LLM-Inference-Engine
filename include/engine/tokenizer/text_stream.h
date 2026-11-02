#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "engine/tokenizer/tokenizer.h"

namespace engine {

using Decode = std::function<std::string(std::span<const int32_t> ids)>;

class TextStream {
public:
    explicit TextStream(const Tokenizer& tokenizer)
        : decode_([&tokenizer](std::span<const int32_t> ids) { return tokenizer.decode(ids); }) {}
    explicit TextStream(Decode decode) : decode_(std::move(decode)) {}

    std::string push(int32_t token);

private:
    Decode decode_;
    std::vector<int32_t> ids_;
    size_t emitted_ = 0;
};

}  // namespace engine
