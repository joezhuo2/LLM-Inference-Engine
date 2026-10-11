#include "engine/kv/block_manager.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace engine {

BlockManager::BlockManager(int num_blocks, int block_size)
    : num_blocks_(num_blocks),
      block_size_(block_size),
      free_count_(num_blocks - 1),
      free_blocks_(num_blocks, true) {
    if (num_blocks < 2 || block_size < 1) {
        throw std::invalid_argument(
            "[BlockManager] constructor parameters num_blocks must be >= 2 and block_size must be "
            ">= 1. "
            "Got num_blocks=" +
            std::to_string(num_blocks_) + ", block_size=" + std::to_string(block_size_));
    }

    free_blocks_[0] = false;
}

int BlockManager::allocate_block() {
    for (int i = 1; i < num_blocks_; ++i) {
        if (free_blocks_[i]) {
            free_blocks_[i] = false;
            free_count_--;
            return i;
        }
    }
    throw std::runtime_error("[BlockManager] Out of free blocks despite prior check.");
}

void BlockManager::free_block(int block_id) {
    if (block_id >= 0 && block_id < num_blocks_ && !free_blocks_[block_id]) {
        free_blocks_[block_id] = true;
        free_count_++;
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
        throw std::runtime_error(
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
        throw std::runtime_error("[BlockManager] out of memory. Cannot append token to sequence " +
                                 std::to_string(seq));
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

    for (int block_id : seq_data.table) {
        free_block(block_id);
    }

    sequences_.erase(seq);
}

int BlockManager::slot(SeqId seq, int pos) const {
    auto& seq_data = get_sequence(seq);

    if (pos < 0 || pos >= seq_data.length) {
        throw std::out_of_range("[BlockManager] slot position out of bounds: " +
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

}  // namespace engine
