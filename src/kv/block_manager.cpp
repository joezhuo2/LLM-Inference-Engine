#include "engine/kv/block_manager.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine {

namespace {

int checked_num_blocks(int num_blocks, int block_size) {
    if (num_blocks < 2 || block_size < 1) {
        throw std::invalid_argument(
            "[BlockManager] num_blocks must be >= 2 and block_size must be >= 1. Got num_blocks=" +
            std::to_string(num_blocks) + ", block_size=" + std::to_string(block_size));
    }
    return num_blocks;
}

std::size_t hash_combine_token(std::size_t seed, int32_t token) noexcept {
    std::uint64_t x = static_cast<std::uint64_t>(static_cast<uint32_t>(token));
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);

    return seed ^ (static_cast<size_t>(x) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

}  // namespace

BlockManager::BlockManager(int num_blocks, int block_size)
    : num_blocks_(checked_num_blocks(num_blocks, block_size)),
      block_size_(block_size),
      free_count_(num_blocks_ - 1),
      free_blocks_(num_blocks_, true),
      block_meta_(checked_num_blocks(num_blocks, block_size)) {
    free_blocks_[0] = false;
}

void BlockManager::evict_block(int id) {
    auto& meta = block_meta_[id];

    if (meta.published) {
        lru_cached_.erase(meta.lru_it);
        prefix_to_block_.erase(meta.key);
        meta.published = false;
        meta.key = PrefixKey{};
    }

    free_blocks_[id] = true;
}

int BlockManager::allocate_block() {
    for (int i = 1; i < num_blocks_; ++i) {
        if (free_blocks_[i]) {
            free_blocks_[i] = false;
            free_count_--;
            block_meta_[i].ref_count = 1;
            return i;
        }
    }

    if (!lru_cached_.empty()) {
        int victim = lru_cached_.front();
        evict_block(victim);
        free_blocks_[victim] = false;
        free_count_--;
        block_meta_[victim].ref_count = 1;
        return victim;
    }

    throw std::runtime_error("[BlockManager] Out of free blocks despite prior check.");
}

void BlockManager::free_block(int block_id) {
    if (block_id <= 0 || block_id >= num_blocks_) {
        return;
    }

    auto& meta = block_meta_[block_id];

    if (meta.ref_count <= 0) {
        return;
    }

    meta.ref_count--;

    if (meta.ref_count == 0) {
        free_count_++;

        if (meta.published) {
            meta.lru_it = lru_cached_.insert(lru_cached_.end(), block_id);
        } else {
            free_blocks_[block_id] = true;
        }
    }
}

const BlockManager::Sequence& BlockManager::get_sequence(SeqId seq) const {
    auto it = sequences_.find(seq);
    if (it == sequences_.end()) {
        throw std::invalid_argument("[BlockManager] Sequence id " + std::to_string(seq) +
                                    " is not registered.");
    }
    return it->second;
}

BlockManager::Sequence& BlockManager::get_sequence(SeqId seq) {
    auto it = sequences_.find(seq);
    if (it == sequences_.end()) {
        throw std::invalid_argument("[BlockManager] Sequence id " + std::to_string(seq) +
                                    " is not registered.");
    }
    return it->second;
}

bool BlockManager::can_allocate(int num_tokens) const {
    if (num_tokens < 1) {
        throw std::invalid_argument("[BlockManager] num_tokens must be >= 1. Got num_tokens=" +
                                    std::to_string(num_tokens));
    }

    int blocks_needed = (num_tokens + block_size_ - 1) / block_size_;

    return free_count_ >= blocks_needed;
}

void BlockManager::allocate(SeqId seq, int num_tokens) {
    if (num_tokens < 1) {
        throw std::invalid_argument("[BlockManager] num_tokens must be >= 1. Got num_tokens=" +
                                    std::to_string(num_tokens));
    }

    if (sequences_.find(seq) != sequences_.end()) {
        throw std::invalid_argument("[BlockManager] sequence id " + std::to_string(seq) +
                                    " is already registered.");
    }

    if (!can_allocate(num_tokens)) {
        throw std::invalid_argument(
            "[BlockManager] out of memory. Insufficient free blocks for sequence " +
            std::to_string(seq) + " (requested tokens: " + std::to_string(num_tokens) + ")");
    }

    int blocks_needed = (num_tokens + block_size_ - 1) / block_size_;
    Sequence seq_data;
    seq_data.length = num_tokens;

    for (int i = 0; i < blocks_needed; ++i) {
        seq_data.table.push_back(allocate_block());
    }

    sequences_[seq] = std::move(seq_data);
}

bool BlockManager::can_append(SeqId seq) const {
    const auto& seq_data = get_sequence(seq);
    if (seq_data.length % block_size_ == 0) {
        return free_count_ >= 1;
    }
    return true;
}

int BlockManager::append_slot(SeqId seq) {
    if (!can_append(seq)) {
        throw std::invalid_argument(
            "[BlockManager] out of memory. Cannot append token to sequence " + std::to_string(seq));
    }

    auto& seq_data = get_sequence(seq);

    if (seq_data.length % block_size_ == 0) {
        int new_block = allocate_block();
        seq_data.table.push_back(new_block);
    }

    int pos = seq_data.length;
    seq_data.length++;

    return seq_data.table.back() * block_size_ + pos % block_size_;
}

void BlockManager::free(SeqId seq) {
    auto& seq_data = get_sequence(seq);

    for (auto it = seq_data.table.rbegin(); it != seq_data.table.rend(); ++it) {
        free_block(*it);
    }

    sequences_.erase(seq);
}

void BlockManager::truncate(SeqId seq, int length) {
    auto& seq_data = get_sequence(seq);

    if (length < 1 || length > seq_data.length) {
        throw std::invalid_argument(
            "[BlockManager] invalid truncate length " + std::to_string(length) + " for sequence " +
            std::to_string(seq) + " of length " + std::to_string(seq_data.length));
    }

    if (length == seq_data.length) {
        return;
    }

    int new_blocks = (length + block_size_ - 1) / block_size_;

    while (static_cast<int>(seq_data.table.size()) > new_blocks) {
        free_block(seq_data.table.back());
        seq_data.table.pop_back();
    }

    seq_data.length = length;
}

int BlockManager::slot(SeqId seq, int pos) const {
    auto& seq_data = get_sequence(seq);

    if (pos < 0 || pos >= seq_data.length) {
        throw std::invalid_argument("[BlockManager] slot position out of bounds: " +
                                    std::to_string(pos) + " for sequence " + std::to_string(seq) +
                                    " of length " + std::to_string(seq_data.length));
    }

    return seq_data.table[pos / block_size_] * block_size_ + pos % block_size_;
}

const std::vector<int>& BlockManager::table(SeqId seq) const {
    return get_sequence(seq).table;
}

int BlockManager::length(SeqId seq) const {
    return get_sequence(seq).length;
}

int BlockManager::num_free() const {
    return free_count_;
}

int BlockManager::match_blocks(std::span<const int32_t> tokens,
                               std::vector<int>* matched_out) const {
    if (tokens.empty()) {
        throw std::invalid_argument("[BlockManager] tokens must not be empty.");
    }

    const int max_matchable = static_cast<int>((tokens.size() - 1) / block_size_);

    if (max_matchable <= 0) {
        return 0;
    }

    PrefixKey key;
    key.hash = 0;
    key.tokens.reserve(static_cast<std::size_t>(max_matchable) * block_size_);

    int matched = 0;

    for (int b = 0; b < max_matchable; ++b) {
        const int start = b * block_size_;

        for (int j = 0; j < block_size_; ++j) {
            int32_t tok = tokens[start + j];
            key.hash = hash_combine_token(key.hash, tok);
            key.tokens.push_back(tok);
        }

        auto it = prefix_to_block_.find(key);

        if (it == prefix_to_block_.end()) {
            break;
        }

        ++matched;

        if (matched_out != nullptr) {
            matched_out->push_back(it->second);
        }
    }

    return matched;
}

int BlockManager::match_prefix(std::span<const int32_t> tokens) const {
    return match_blocks(tokens, nullptr) * block_size_;
}

bool BlockManager::can_allocate_tokens(std::span<const int32_t> tokens) const {
    std::vector<int> matched;
    int leading = match_blocks(tokens, &matched);

    int cached = 0;

    for (int block_id : matched) {
        if (block_meta_[block_id].ref_count == 0) {
            ++cached;
        }
    }

    const int total = (static_cast<int>(tokens.size()) + block_size_ - 1) / block_size_;
    const int needed = total - leading;
    const int available = free_count_ - cached;

    return available >= needed;
}

int BlockManager::allocate_tokens(SeqId seq, std::span<const int32_t> tokens) {
    if (tokens.empty()) {
        throw std::invalid_argument("[BlockManager] tokens must not be empty.");
    }

    if (sequences_.find(seq) != sequences_.end()) {
        throw std::invalid_argument("[BlockManager] sequence id " + std::to_string(seq) +
                                    " is already registered.");
    }

    std::vector<int> matched;
    int leading = match_blocks(tokens, &matched);

    int cached = 0;

    for (int block_id : matched) {
        if (block_meta_[block_id].ref_count == 0) {
            ++cached;
        }
    }

    const int total = (static_cast<int>(tokens.size()) + block_size_ - 1) / block_size_;
    const int needed = total - leading;
    const int available = free_count_ - cached;

    if (available < needed) {
        throw std::invalid_argument(
            "[BlockManager] out of memory. Insufficient free blocks for sequence " +
            std::to_string(seq) + " (requested tokens: " + std::to_string(tokens.size()) + ")");
    }

    for (int block_id : matched) {
        auto& meta = block_meta_[block_id];
        if (meta.ref_count == 0) {
            lru_cached_.erase(meta.lru_it);
            --free_count_;
        }

        ++meta.ref_count;
    }

    Sequence seq_data;
    seq_data.length = static_cast<int>(tokens.size());
    seq_data.table.reserve(total);
    seq_data.table.insert(seq_data.table.end(), matched.begin(), matched.end());

    for (int i = 0; i < needed; ++i) {
        seq_data.table.push_back(allocate_block());
    }

    sequences_[seq] = std::move(seq_data);
    return leading * block_size_;
}

void BlockManager::publish(SeqId seq, std::span<const int32_t> tokens) {
    const auto& seq_data = get_sequence(seq);

    if (static_cast<int>(tokens.size()) > seq_data.length) {
        throw std::invalid_argument("[BlockManager] Token count (" + std::to_string(tokens.size()) +
                                    ") does not match sequence length (" +
                                    std::to_string(seq_data.length) + ")");
    }

    const int full_blocks = static_cast<int>(tokens.size()) / block_size_;

    if (full_blocks <= 0) {
        return;
    }

    PrefixKey key;
    key.hash = 0;
    key.tokens.reserve(static_cast<std::size_t>(full_blocks) * block_size_);

    for (int b = 0; b < full_blocks; ++b) {
        const int start = b * block_size_;

        for (int j = 0; j < block_size_; ++j) {
            const int32_t tok = tokens[start + j];
            key.hash = hash_combine_token(key.hash, tok);
            key.tokens.push_back(tok);
        }

        const int block_id = seq_data.table[b];
        auto& meta = block_meta_[block_id];

        if (meta.published) {
            continue;
        }

        auto [it, inserted] = prefix_to_block_.emplace(key, block_id);

        if (inserted) {
            meta.published = true;
            meta.key = key;
        }
    }
}

int BlockManager::ref_count(int block) const {
    if (block < 0 || block >= num_blocks_) {
        throw std::invalid_argument("[BlockManager] block id out of bounds: " +
                                    std::to_string(block));
    }

    if (block == 0) {
        return 0;
    }

    return block_meta_[block].ref_count;
}

int BlockManager::num_cached() const {
    return static_cast<int>(lru_cached_.size());
}

}  // namespace engine
