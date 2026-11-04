#pragma once

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "llm/config.h"
#include "llm/device_buffer.h"
#include "llm/kv_cache.h"
#include "llm/loader.h"

namespace llm {

// Free VRAM an auto-sized KV pool leaves untouched: other processes and the display share the GPU.
constexpr std::size_t kKVHeadroomBytes = std::size_t{256} << 20;

// A matmul weight [N, K] in any stored format; scales/mins are [N, K / group_size]
struct QuantWeight {
  DType dtype = DType::kFP16;
  const void* data = nullptr;
  const __half* scales = nullptr;  // INT8 / INT4 only
  const __half* mins = nullptr;    // INT4 only
  int group_size = 0;              // 0 for FP16

  const __half* fp16() const { return static_cast<const __half*>(data); }
};

struct LayerWeights {
  const __half* attn_norm = nullptr;  // [hidden]
  QuantWeight wq;                     // [num_heads * head_dim, hidden]
  QuantWeight wk;                     // [num_kv_heads * head_dim, hidden]
  QuantWeight wv;                     // [num_kv_heads * head_dim, hidden]
  QuantWeight wo;                     // [hidden, num_heads * head_dim]
  const __half* ffn_norm = nullptr;   // [hidden]
  QuantWeight gate_proj;              // [intermediate, hidden]
  QuantWeight up_proj;                // [intermediate, hidden]
  QuantWeight down_proj;              // [hidden, intermediate]
};

struct Activations {
  int max_tokens = 0;

  DeviceBuffer ids;        // int32  [max_tokens]
  DeviceBuffer positions;  // int32  [max_tokens]
  DeviceBuffer row_slots;  // int32  [max_tokens]                cache slot of each row

  DeviceBuffer x;          // float  [max_tokens, hidden]        residual stream
  DeviceBuffer normed;     // float  [max_tokens, hidden]        norm output
  DeviceBuffer q;          // float  [max_tokens, num_heads * head_dim]
  DeviceBuffer k;          // float  [max_tokens, num_kv_heads * head_dim]
  DeviceBuffer v;          // float  [max_tokens, num_kv_heads * head_dim]
  DeviceBuffer scores;     // float  [num_heads, max_tokens, max_tokens]
  DeviceBuffer attn_out;   // float  [max_tokens, num_heads * head_dim]
  DeviceBuffer proj;       // float  [max_tokens, hidden]        wo / down_proj output
  DeviceBuffer gate;       // float  [max_tokens, intermediate]
  DeviceBuffer up;         // float  [max_tokens, intermediate]
  DeviceBuffer act;        // float  [max_tokens, intermediate]  swiglu output
  DeviceBuffer logits;     // float  [max_tokens, vocab]
  DeviceBuffer next_token; // int32  [max_tokens]               argmax of each logits row

  void allocate(const ModelConfig& config, int max_tokens);
  std::size_t total_bytes() const;
};

class Model {
 public:
  // max_tokens bounds each sequence; max_seqs is how many sequences the cache holds at once.
  // kv_pages sizes the shared page pool; 0 fits it to free VRAM, capped at what the slots can use.
  static Model load(const std::string& path, int max_tokens, int max_seqs = 1, int kv_pages = 0);

  const ModelConfig& config() const { return config_; }
  std::size_t weight_bytes() const { return weights_.bytes(); }
  std::size_t workspace_bytes() const { return act_.total_bytes(); }
  std::size_t cache_bytes() const { return cache_.bytes(); }
  int max_tokens() const { return act_.max_tokens; }
  int max_seqs() const { return cache_.max_seqs(); }
  int kv_pages() const { return cache_.pool().num_pages(); }

  // Sequences of max_tokens that can decode at once: limited by slots or by pages.
  int max_batch_size() const {
    return std::min(max_seqs(), kv_pages() / cache_.max_pages_per_seq());
  }

  // Runs n_tokens of one sequence, appending them to that sequence's cache slot.
  void forward(const std::int32_t* host_ids, int n_tokens, int slot = 0);

  // One decode step for several sequences: host_ids[b] is the next token of the one in slots[b].
  // Logits row b belongs to slots[b].
  void forward_decode(const std::int32_t* host_ids, const int* slots, int n_seqs);

  void reset_cache() { cache_.reset_all(); }

  // Empties one sequence's slot and returns its pages to the pool.
  void release_slot(int slot) { cache_.reset(slot); }

  const KVCache& cache() const { return cache_; }
  void shuffle_cache_pages(std::uint32_t seed) { cache_.shuffle_free_pages(seed); }

  // Copies the logits of the last forward pass back to the host.
  std::vector<float> logits_host(int n_tokens);

  // Copies one logits row of the last forward pass back to the host.
  std::vector<float> logits_row(int row);

  std::int32_t argmax_last(int n_tokens);

  // Next token of each of the first n_rows logits rows, e.g. one per sequence after forward_decode.
  std::vector<std::int32_t> argmax_rows(int n_rows);

  // Called once per generated token, as soon as its id is on the host. `step` is
  // 0 for the first token. 
  using TokenCallback = std::function<void(int step, std::int32_t token)>;

  std::vector<std::int32_t> generate(const std::vector<std::int32_t>& prompt, int max_new_tokens,
                                     const TokenCallback& on_token = nullptr);

  // Like TokenCallback, with `seq` indexing the prompts passed to generate_batch.
  using BatchTokenCallback = std::function<void(int seq, int step, std::int32_t token)>;

  // Greedy-decodes every prompt, decoding up to max_seqs of them together. A prompt waits, in
  // order, until a slot and the pages for its whole run are free. Returns ids per prompt.
  std::vector<std::vector<std::int32_t>> generate_batch(
      const std::vector<std::vector<std::int32_t>>& prompts, int max_new_tokens,
      const BatchTokenCallback& on_token = nullptr);

 private:
  ModelConfig config_;
  DeviceBuffer weights_;  // one allocation holding every tensor's bytes
  Activations act_;
  KVCache cache_;

  const __half* tok_embeddings_ = nullptr;  // [vocab, hidden]
  const __half* output_norm_ = nullptr;     // [hidden]
  QuantWeight lm_head_;                     // [vocab, hidden]
  std::vector<LayerWeights> layers_;

  // Runs one row per entry of row_slots; rows of the same sequence must be adjacent.
  void forward_rows(const std::int32_t* host_ids, const std::vector<int>& row_slots);

  int q_dim() const { return static_cast<int>(config_.num_heads * config_.head_dim); }
  int kv_dim() const { return static_cast<int>(config_.num_kv_heads * config_.head_dim); }
};

}  // namespace llm
