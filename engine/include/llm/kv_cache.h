#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "llm/allocator.h"
#include "llm/config.h"
#include "llm/device_buffer.h"
#include "llm/kernels.h"

namespace llm {

constexpr int kKVTokenShift = 6;
constexpr int kKVTokensPerPage = 1 << kKVTokenShift;

// K and V rows for up to max_seqs sequences, in pages taken from one shared pool as they grow.
// A page holds K for every layer, then V for every layer, for kKVTokensPerPage tokens.
// Each sequence lives in a slot with its own page list, length and row of the page table.
class KVCache {
 public:
  void allocate(const ModelConfig& config, int max_tokens, int max_seqs = 1);

  // Makes sure the slot's next n_new tokens have pages; call before writing them.
  void reserve(int slot, int n_new);

  // Appends n_new rows of [kv_dim] to one layer of the slot, starting at length(slot).
  void write(int slot, int layer, const float* new_k, const float* new_v, int n_new);

  KVPages k_pages(int slot, int layer) const;
  KVPages v_pages(int slot, int layer) const;

  int length(int slot) const { return slot_at(slot).length; }

  void advance(int slot, int n_new);

  // Returns the slot's pages to the pool.
  void reset(int slot);
  void reset_all();

  int max_tokens() const { return max_tokens_; }  // per sequence
  int max_seqs() const { return static_cast<int>(slots_.size()); }
  int max_pages_per_seq() const { return max_pages_per_seq_; }
  int kv_dim() const { return kv_dim_; }
  std::size_t bytes() const { return pool_.total_bytes() + page_table_.bytes(); }
  const PagePool& pool() const { return pool_; }

  // Device int32 [max_seqs, max_pages_per_seq]; row s is slot s's page ids in token order.
  const std::int32_t* page_table() const { return page_table_.as<std::int32_t>(); }

  // Page ids in token order; scattered once shuffle_free_pages has run.
  const std::vector<std::int32_t>& page_ids(int slot) const { return slot_at(slot).pages; }
  void shuffle_free_pages(std::uint32_t seed) { pool_.shuffle_free_list(seed); }

 private:
  struct Slot {
    int length = 0;
    std::vector<std::int32_t> pages;  // host copy of this slot's page-table row
  };

  const Slot& slot_at(int slot) const;
  Slot& slot_at(int slot);
  std::size_t layer_floats() const;
  KVPages pages_at(int slot, int layer) const;

  int max_tokens_ = 0;
  int max_pages_per_seq_ = 0;
  int num_layers_ = 0;
  int kv_dim_ = 0;

  PagePool pool_;
  std::vector<Slot> slots_;
  DeviceBuffer page_table_;
};

}  // namespace llm
