#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "engine/tokenizer/tokenizer.h"

namespace engine {

class TextStream {
public:
    explicit TextStream(const Tokenizer& tokenizer) : tokenizer_(&tokenizer) {}

    std::string push(int32_t token);

private:
    const Tokenizer* tokenizer_;
    std::vector<int32_t> ids_;
    size_t emitted_ = 0;
};

}  // namespace engine
