#include "engine/tokenizer/tokenizer.h"

#include <sentencepiece_processor.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace engine {

namespace {

[[noreturn]] void fail(const std::string& what) {
    throw std::runtime_error("tokenizer: " + what);
}

}  // namespace

Tokenizer::Tokenizer(const std::filesystem::path& model)
    : sp_(std::make_unique<sentencepiece::SentencePieceProcessor>()) {
    if (const auto status = sp_->Load(model.string()); !status.ok())
        fail("cannot load " + model.string() + ": " + status.ToString());
    vocab_size_ = sp_->GetPieceSize();
    unk_id_ = sp_->unk_id();
    bos_id_ = sp_->bos_id();
    eos_id_ = sp_->eos_id();
    if (bos_id_ < 0 || eos_id_ < 0) fail(model.string() + " has no BOS or EOS piece");
    const auto ids = sp_->EncodeAsIds("\n");
    sentinel_.assign(ids.begin(), ids.end());
    if (sentinel_.size() != 2 || sp_->IdToPiece(sentinel_[0]) != "▁" ||
        sp_->IdToPiece(sentinel_[1]) != "<0x0A>")
        fail(model.string() + " must encode '\\n' as the dummy prefix and the <0x0A> byte piece");
}

Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

void Tokenizer::append_text(std::vector<int32_t>& out, std::string_view text, bool first) const {
    std::string input = "\n";
    if (first && text.front() != ' ') input += ' ';
    input += text;
    const auto ids = sp_->EncodeAsIds(input);
    if (ids.size() < 2 || !std::equal(sentinel_.begin(), sentinel_.end(), ids.begin()))
        fail("the newline sentinel merged with the text");
    out.insert(out.end(), ids.begin() + 2, ids.end());
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool add_bos) const {
    const std::array<std::pair<std::string, int32_t>, 3> specials{{
        {sp_->IdToPiece(unk_id_), unk_id_},
        {sp_->IdToPiece(bos_id_), bos_id_},
        {sp_->IdToPiece(eos_id_), eos_id_},
    }};
    std::vector<int32_t> out;
    if (add_bos) out.push_back(bos_id_);
    size_t pos = 0;
    while (pos <= text.size()) {
        size_t next = std::string_view::npos;
        const std::pair<std::string, int32_t>* hit = nullptr;
        for (const auto& s : specials) {
            const size_t at = text.find(s.first, pos);
            if (at < next) {
                next = at;
                hit = &s;
            }
        }
        const std::string_view segment = text.substr(pos, next - pos);
        if (!segment.empty()) append_text(out, segment, pos == 0);
        if (!hit) break;
        out.push_back(hit->second);
        pos = next + hit->first.size();
    }
    return out;
}

std::string Tokenizer::decode(std::span<const int32_t> ids) const {
    std::vector<int> kept;
    kept.reserve(ids.size());
    for (const int32_t id : ids) {
        if (id < 0 || id >= vocab_size_)
            throw std::out_of_range("tokenizer: token id " + std::to_string(id) +
                                    " is out of range");
        if (id != unk_id_ && id != bos_id_ && id != eos_id_) kept.push_back(id);
    }
    return sp_->DecodeIds(kept);
}

}  // namespace engine
