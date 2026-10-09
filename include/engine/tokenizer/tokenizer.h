#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sentencepiece {
class SentencePieceProcessor;
}

namespace engine {

class Tokenizer {
public:
    explicit Tokenizer(const std::filesystem::path& model);
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    std::vector<int32_t> encode(std::string_view text, bool add_bos) const;
    std::string decode(std::span<const int32_t> ids) const;

    int32_t vocab_size() const { return vocab_size_; }
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }

private:
    void append_text(std::vector<int32_t>& out, std::string_view text, bool first) const;

    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_;
    int32_t vocab_size_ = 0;
    int32_t unk_id_ = 0;
    int32_t bos_id_ = 0;
    int32_t eos_id_ = 0;
    std::vector<int32_t> sentinel_;
};

}  // namespace engine
