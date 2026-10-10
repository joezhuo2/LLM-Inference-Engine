#pragma once

#include <cstdint>
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
};

}  // namespace engine
