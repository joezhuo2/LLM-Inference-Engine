#include "engine/kv/block_manager.h"

#include <stdexcept>

namespace engine {

/*
TODO(JOE): BlockManager

Purpose. The block manager owns the bookkeeping of the paged KV cache: it splits the cache into fixed-size blocks of block_size token slots, hands blocks to sequences as they grow, takes them back when sequences finish, and translates a sequence's token position into the physical slot that holds that token's K and V. It exists so that many sequences of unknown final length can share one preallocated cache without reserving the maximum context for each of them (the fragmentation a contiguous per-sequence cache causes), and so that exactly one place decides which slot belongs to which token. The scheduler asks it whether a request fits before admitting it and whether every running sequence can grow before a decode step, and the model runner reads block tables, slots and lengths from it to fill block_tables, slot_mapping and context_lens. It is pure C++ with no CUDA so it can be tested exhaustively on the CPU; it never touches the cache memory itself.

Blocks and slots. Physical blocks are numbered 0 to num_blocks - 1, and block p holds the slots p * block_size to p * block_size + block_size - 1. Block 0 is reserved as the null block and is never given to a sequence, so padded batch entries in later branches can point at it safely; the usable capacity is num_blocks - 1 blocks. A sequence of length L owns exactly ceil(L / block_size) blocks, listed in logical order in its block table, and the token at position pos (0-based) lives in slot table[pos / block_size] * block_size + pos % block_size. A sequence takes a new block only when its length crosses a multiple of block_size, and never holds a block it is not using.

Interface. BlockManager(num_blocks, block_size) starts with every block except block 0 free and no sequences; it requires num_blocks >= 2 and block_size >= 1. can_allocate(num_tokens) says whether ceil(num_tokens / block_size) blocks are free right now. allocate(seq, num_tokens) registers a new sequence with length num_tokens and gives it that many blocks; these slots are where the prompt's K and V will be written. can_append(seq) says whether the sequence can grow by one token, which is true when its current length is not a multiple of block_size (the last block has room) or when at least one block is free. append_slot(seq) grows the sequence by one token, taking a new block exactly when the old length is a multiple of block_size, and returns the slot of the new token, which is the token at position old length; afterwards slot(seq, length(seq) - 1) returns the same value. free(seq) returns all of the sequence's blocks to the free pool and forgets the sequence, so its id may be allocated again. slot(seq, pos) returns the slot of position pos. table(seq) returns the sequence's block table (physical block ids in logical order); the reference is valid until the next non-const call. length(seq) returns the number of tokens the sequence holds. num_free() returns the number of free blocks. SeqId is a caller-chosen 64-bit id (the request id); the block manager does not interpret it. The private representation is yours to choose; the public interface above is what the tests and later branches use.

Errors. Every violated precondition throws std::invalid_argument with a message that names the offending sequence id or argument, and leaves the block manager exactly as it was before the call (no blocks taken, no lengths changed, no sequence added or removed). The preconditions are: num_blocks >= 2 and block_size >= 1 for the constructor; num_tokens >= 1 for can_allocate and allocate; seq not already registered for allocate; seq registered for every other call that takes a seq; enough free blocks for allocate (the caller is expected to ask can_allocate first); can_append(seq) for append_slot; 0 <= pos < length(seq) for slot. std::invalid_argument is required, not just any std::logic_error, so a rejection test cannot pass against this stub.

Invariants. At every point between calls: no block is owned by two sequences; block 0 is owned by nobody and never free; num_free() plus the total number of blocks in all tables equals num_blocks - 1; every table has exactly ceil(length / block_size) entries; and the slots of all positions of all sequences are pairwise distinct. The result is deterministic: two block managers given the same sequence of calls hand out the same block ids. Which free block is handed out next is otherwise unspecified, and nothing may assume that a sequence's blocks are contiguous or increasing.

Edge cases and boundaries. With block_size 16, lengths 15, 16 and 17 own 1, 1 and 2 blocks; appending to a sequence of length 15 or 17 takes no block, appending to one of length 16 takes one. can_allocate must say no once ceil(num_tokens / block_size) exceeds num_free(), including when the cache is exactly full; a full cache must still let sequences whose last block has room keep appending. A length of 2,048 (the MVP context limit, 128 blocks at block_size 16) must work like any other; enforcing that limit is the scheduler's job, not this class's. Sizes that would overflow int are out of scope.

What to compare against. A shadow model in the test that tracks each sequence's length and checks the invariants and the slot formula after every call.

Tests that prove it. tests/todo/unit/test_block_manager.cpp: constructor checks, the block counts at 15, 16, 17 and 2,048, block 0 never handed out, the slot formula, append taking a block only at multiples of block_size, a full cache, freeing and reusing an id, every rejected call with the state unchanged afterwards, determinism, and 10,000 random allocate, append and free operations checked against the shadow model after each one. They run in the todo_unit_tests target with the todo_joe label (ctest --preset debug -L todo_joe). When they all pass, move the file to tests/unit/, add it to unit_tests and delete todo_unit_tests if it is empty.
*/
BlockManager::BlockManager(int, int) {
    throw std::logic_error("unimplemented: BlockManager");
}

bool BlockManager::can_allocate(int) const {
    throw std::logic_error("unimplemented: BlockManager");
}

void BlockManager::allocate(SeqId, int) {
    throw std::logic_error("unimplemented: BlockManager");
}

bool BlockManager::can_append(SeqId) const {
    throw std::logic_error("unimplemented: BlockManager");
}

int BlockManager::append_slot(SeqId) {
    throw std::logic_error("unimplemented: BlockManager");
}

void BlockManager::free(SeqId) {
    throw std::logic_error("unimplemented: BlockManager");
}

int BlockManager::slot(SeqId, int) const {
    throw std::logic_error("unimplemented: BlockManager");
}

const std::vector<int>& BlockManager::table(SeqId) const {
    throw std::logic_error("unimplemented: BlockManager");
}

int BlockManager::length(SeqId) const {
    throw std::logic_error("unimplemented: BlockManager");
}

int BlockManager::num_free() const {
    throw std::logic_error("unimplemented: BlockManager");
}

}  // namespace engine
