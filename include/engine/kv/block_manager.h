#pragma once

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace engine {

using SeqId = int64_t;

class BlockManager {
public:
    BlockManager(int num_blocks, int block_size);

    bool can_allocate(int num_tokens) const;
    void allocate(SeqId seq, int num_tokens);
    bool can_append(SeqId seq) const;
    int append_slot(SeqId seq);
    void free(SeqId seq);

    int slot(SeqId seq, int pos) const;
    const std::vector<int>& table(SeqId seq) const;
    int length(SeqId seq) const;
    int num_free() const;

    int match_prefix(std::span<const int32_t> tokens) const;
    bool can_allocate_tokens(std::span<const int32_t> tokens) const;
    int allocate_tokens(SeqId seq, std::span<const int32_t> tokens);
    void publish(SeqId seq, std::span<const int32_t> tokens);
    int ref_count(int block) const;
    int num_cached() const;

private:
    struct Sequence {
        int length;
        std::vector<int> table;
    };

    int num_blocks_;
    int block_size_;
    int free_count_;
    std::vector<bool> free_blocks_;
    std::unordered_map<SeqId, Sequence> sequences_;

    int allocate_block();
    void free_block(int block_id);
    const Sequence& get_sequence(SeqId seq) const;
    Sequence& get_sequence(SeqId seq);
};

}  // namespace engine
