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

// One sequence's K and V rows, stored in pages taken from a pool as it grows.
// A page holds K for every layer, then V for every layer, for kKVTokensPerPage tokens.
class KVCache {
 public:
  void allocate(const ModelConfig& config, int max_tokens);

  // Makes sure the next n_new tokens have pages; call before writing them.
  void reserve(int n_new);

  // Appends n_new rows of [kv_dim] to one layer, starting at length().
  void write(int layer, const float* new_k, const float* new_v, int n_new);

  KVPages k_pages(int layer) const;
  KVPages v_pages(int layer) const;

  int length() const { return length_; }

  void advance(int n_new);

  // Returns every page to the pool.
  void reset();

  int max_tokens() const { return max_tokens_; }
  int kv_dim() const { return kv_dim_; }
  std::size_t bytes() const { return pool_.total_bytes() + page_table_.bytes(); }
  const PagePool& pool() const { return pool_; }

 private:
  std::size_t layer_floats() const;
  KVPages pages_at(int layer) const;

  int max_tokens_ = 0;
  int num_layers_ = 0;
  int kv_dim_ = 0;
  int length_ = 0;

  PagePool pool_;
  std::vector<std::int32_t> pages_;  // host copy of the page table
  DeviceBuffer page_table_;          // int32 [pool pages], page id for each run of tokens
};

}  // namespace llm
