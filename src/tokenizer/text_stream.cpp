#include "engine/tokenizer/text_stream.h"

#include <string_view>

namespace engine {

namespace {

constexpr std::string_view kReplacementChar = "\xEF\xBF\xBD";

}  // namespace

std::string TextStream::push(int32_t token) {
    ids_.push_back(token);
    const std::string text = tokenizer_->decode(ids_);
    if (text.ends_with(kReplacementChar) || text.size() <= emitted_) return {};
    std::string out = text.substr(emitted_);
    emitted_ = text.size();
    return out;
}

}  // namespace engine
