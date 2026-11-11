#include "engine/kv/block_manager.h"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

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

}  // namespace

BlockManager::BlockManager(int num_blocks, int block_size)
    : num_blocks_(checked_num_blocks(num_blocks, block_size)),
      block_size_(block_size),
      free_count_(num_blocks_ - 1),
      free_blocks_(num_blocks_, true) {
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

    for (int block_id : seq_data.table) {
        free_block(block_id);
    }

    sequences_.erase(seq);
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

/*
TODO(JOE): prefix caching (match_prefix, can_allocate_tokens, allocate_tokens, publish, ref_count, num_cached)

Purpose. Issue #99. Today every request computes the keys and values of its whole prompt, even when the first thousand tokens are the same chat template and system prompt as the request before it. The keys and values of a token depend only on the tokens up to it, so two sequences that start with the same tokens can read the same physical blocks. These six functions let the block manager find the full blocks a new sequence can reuse, count how many sequences hold each block, and keep the blocks of a freed sequence around until the memory is needed. The scheduler then starts the prefill of a request at the first token that was not found. Nothing changes on the device: a block table row may simply name a block that another row also names.

Words. A block is blank when no sequence holds it and it has no published content, in use when at least one sequence holds it, and cached when no sequence holds it but it is published. A block is published once publish has recorded which tokens its keys and values were computed for: all the tokens of the sequence from position 0 to the end of that block. Only full blocks are ever published.

Interface. Declared in include/engine/kv/block_manager.h. match_prefix(tokens) returns how many leading tokens of tokens can be reused right now: block_size * b, where b is the largest number such that for every i < b there is a published block for tokens[0 .. (i + 1) * block_size), and b <= (tokens.size() - 1) / block_size. The last token is never matched, because the model has to run on at least one row to produce the logits the first new token is sampled from. match_prefix is const and does not change the eviction order. can_allocate_tokens(tokens) says whether allocate_tokens would succeed. allocate_tokens(seq, tokens) registers seq with length tokens.size(): the first b entries of its table are the matched blocks, in order, and the rest are fresh blocks; it returns block_size * b, the number of tokens whose keys and values are already in the cache. publish(seq, tokens) tells the manager that the keys and values of the first tokens.size() tokens of seq are written and that those tokens are tokens (prompt and generated tokens alike); each full block in that range becomes published. ref_count(block) is the number of registered sequences whose table holds block. num_cached() is the number of cached blocks.

Semantics. Matching is by the whole chain, not by a block's own 16 tokens: a block whose tokens equal another block's but whose earlier tokens differ holds different keys and values and must not match. The plan (docs/claude/build_plan.md, P5) hashes each full block by (hash of the previous block, its token ids) and keeps a hash to block map. A 64-bit hash is not proof: two different prefixes must never match, so keep what you need to compare the tokens on a hit. If you key a block by its parent's physical block id instead, remember that the parent can be evicted and its id reused for other tokens while the child is still in the map. Counting: num_free() becomes blank blocks plus cached blocks, since both can be handed out. A fresh block is the lowest numbered blank block, as today; only when no blank block is left is a cached block evicted (its published record is dropped and it is handed out). Eviction takes the least recently used cached block, where a block's time of use is the moment its count last fell to 0, and the blocks released by one free(seq) are ordered so that the last block of the table is evicted first and the first block last (a chain is only useful from its start). A block matched by an allocate_tokens call is never evicted by that same call. allocate_tokens adds 1 to the count of every matched block (a cached one becomes in use again and leaves the eviction order) and gives each fresh block a count of 1. So with m matched blocks of which z are cached, the call needs ceil(n / block_size) - m fresh blocks out of num_free() - z. free(seq) subtracts 1 from the count of each block of seq; a block that reaches 0 becomes cached if it is published and blank if not. publish skips blocks that are already published; if another block is already published for the same tokens (two requests computed the same prefix side by side), the first one stays in the map and this sequence's block stays unpublished and private. publish may be called again with more tokens as a sequence grows. allocate(seq, num_tokens), can_allocate, append_slot, can_append, slot, table and length keep their meaning, with blocks coming from blank first and evictions second; blocks from allocate are unpublished until publish is called.

Why there is no copy on write here. A sequence writes only into its last block, and only while that block is partial. Partial blocks are never published, so they are never shared, and append_slot on a sequence whose last block is full takes a fresh block. Copy on write is only needed for sharing partial blocks (n > 1 sampling from one prompt), which is not part of this stub.

Errors. Throw std::invalid_argument with a message starting with [BlockManager] when: tokens is empty in match_prefix, can_allocate_tokens or allocate_tokens; seq is already registered or there are not enough blocks in allocate_tokens (nothing may change in that case); seq is not registered or tokens.size() is above length(seq) in publish (an empty tokens there is valid and does nothing); block is below 0 or not below num_blocks in ref_count. ref_count(0) is 0: block 0 stays the reserved block. The caller guarantees that the tokens passed to publish really are the tokens of seq and that their keys and values are fully written (the step that computed them has finished on the device).

Edge cases. tokens of length 1 never match. A prompt of exactly k full blocks matches at most k - 1. block_size 1. A request that matches blocks which are all still in use needs free blocks only for the rest. Preemption gets cheap for free: a preempted sequence whose blocks were published finds them again when it is readmitted, unless they were evicted in between.

What to compare against. The tests keep their own record of which tokens each physical block was published for and check every match against it over a few thousand random allocate, append, publish and free calls. For speed, the plan's done-when: a benchmark where every request shares a 1,000-token system prompt, TTFT on a miss against TTFT on a hit, which should fall by the prefill time of the matched tokens; report the hit rate too.

What is not here. The scheduler side is a separate branch once the tests pass: admission calls can_allocate_tokens and allocate_tokens with the request's tokens, prefill starts at the returned count, publish is called after the step that wrote a block has been updated, and finished and cancelled requests keep going through free. After that a flag on engine serve and engine bench, hit counters in the metrics, shared blocks highlighted on the KV block map and the numbers in docs/results.md.

Tests that prove it. tests/todo/unit/test_prefix_cache.cpp (target todo_unit_tests, label todo_joe), 27 tests checked against a throwaway implementation and eight mutants of it: matching and its limits, sharing and counts, what free keeps, the eviction order, duplicates, generated tokens, block sizes 1 to 32, every rejected argument, and 6,000 random calls per block size against a record of what each block holds. Run them with ctest --preset cpu -L todo_joe. The existing tests in tests/unit/test_block_manager.cpp must keep passing unchanged. When all pass, move the file to tests/unit/, add it to unit_tests and delete todo_unit_tests.
*/
int BlockManager::match_prefix(std::span<const int32_t>) const {
    throw std::logic_error("unimplemented: BlockManager::match_prefix");
}

bool BlockManager::can_allocate_tokens(std::span<const int32_t>) const {
    throw std::logic_error("unimplemented: BlockManager::can_allocate_tokens");
}

int BlockManager::allocate_tokens(SeqId, std::span<const int32_t>) {
    throw std::logic_error("unimplemented: BlockManager::allocate_tokens");
}

void BlockManager::publish(SeqId, std::span<const int32_t>) {
    throw std::logic_error("unimplemented: BlockManager::publish");
}

int BlockManager::ref_count(int) const {
    throw std::logic_error("unimplemented: BlockManager::ref_count");
}

int BlockManager::num_cached() const {
    throw std::logic_error("unimplemented: BlockManager::num_cached");
}

}  // namespace engine
