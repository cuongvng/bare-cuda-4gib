#pragma once

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace llm {

// Finds one layer's K or V rows in a paged cache: token t is at row
// t & (2^token_shift - 1) of page page_table[t >> token_shift].
struct KVPages {
  const float* pool = nullptr;
  const std::int32_t* page_table = nullptr;
  std::size_t page_stride = 0;   // floats from one page to the next
  std::size_t layer_offset = 0;  // floats from a page's start to this layer's rows
  int token_shift = 0;
  std::size_t table_stride = 0;  // entries from one query row's table to the next; 0 shares one
  const std::int32_t* row_slot = nullptr;  // table row each query row reads; null means row r reads r
};

// Puts every token in page 0, so one unbroken buffer can pose as a paged cache.
constexpr int kSinglePageShift = 30;

// Toolchain smoke test: c[i] = a[i] + b[i] over device pointers.
void launch_vector_add(const float* d_a, const float* d_b, float* d_c, std::size_t n,
                       void* stream = nullptr);

// out[t, :] = table[ids[t], :]   -- table is [vocab, hidden], out is [n_tokens, hidden].
void launch_embedding_lookup(const __half* table, const std::int32_t* ids, float* out,
                             int n_tokens, int hidden, void* stream = nullptr);

// Row-wise RMS norm: out[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * weight.
// `x` and `out` are [n_rows, hidden]; `weight` is [hidden]. In-place (out == x)
// is allowed.
void launch_rmsnorm(const float* x, const __half* weight, float* out, int n_rows, int hidden,
                    float eps, void* stream = nullptr);

// Rotary position embedding, applied in place to both q [n_tokens, n_heads, head_dim]
// and k [n_tokens, n_kv_heads, head_dim]. `positions` is one absolute position
// per token (not assumed to be 0..n_tokens-1: a cached decode step passes the
// real position). Pairing is Llama's rotate_half -- element i rotates with
// element i + head_dim/2 -- which is what the HuggingFace checkpoints this
// engine converts are trained with.
void launch_rope(float* q, float* k, const std::int32_t* positions, int n_tokens, int n_heads,
                 int n_kv_heads, int head_dim, float theta, void* stream = nullptr);

// y[n] = sum_k weight[n, k] * x[k]. Weight is row-major [n_out, k_in], i.e. the
// same orientation HuggingFace stores a Linear in, so no transpose pass is
// needed at load time.
void launch_gemv_fp16(const __half* weight, const float* x, float* y, int n_out, int k_in,
                      void* stream = nullptr);

// y[m, n] = sum_k x[m, k] * weight[n, k]. Same weight orientation as the GEMV;
// `x` is [m_rows, k_in] and `y` is [m_rows, n_out].
void launch_gemm_fp16(const __half* weight, const float* x, float* y, int m_rows, int n_out,
                      int k_in, void* stream = nullptr);

// INT8 versions of the two above: weight is int8 [n_out, k_in], scales is FP16
// [n_out, k_in / group_size], and w = q * scale.
// Requires k_in % group_size == 0, group_size % 32 == 0, and a 4-byte-aligned weight.
void launch_gemv_int8(const std::int8_t* weight, const __half* scales, const float* x, float* y,
                      int n_out, int k_in, int group_size, void* stream = nullptr);
void launch_gemm_int8(const std::int8_t* weight, const __half* scales, const float* x, float* y,
                      int m_rows, int n_out, int k_in, int group_size, void* stream = nullptr);

// INT4 versions: weight is [n_out, k_in / 2] bytes, two values per byte with even k in the
// low nibble; scales and mins are FP16 [n_out, k_in / group_size], and w = q * scale + min.
// Same requirements as INT8; the GEMV also needs x 16-byte aligned.
void launch_gemv_int4(const std::uint8_t* weight, const __half* scales, const __half* mins,
                      const float* x, float* y, int n_out, int k_in, int group_size,
                      void* stream = nullptr);
void launch_gemm_int4(const std::uint8_t* weight, const __half* scales, const __half* mins,
                      const float* x, float* y, int m_rows, int n_out, int k_in, int group_size,
                      void* stream = nullptr);

// Appends `n_new` rows of freshly computed K and V into one layer's cache.
// `new_k` and `new_v` are [n_new, kv_dim]; `k_cache` and `v_cache` are that
// layer's [rows, kv_dim] slabs, and row i of the input lands at row
// `start_pos + i`. The caller guarantees start_pos + n_new fits the slab.
void launch_kv_cache_write(const float* new_k, const float* new_v, float* k_cache, float* v_cache,
                           int n_new, int start_pos, int kv_dim, void* stream = nullptr);

// scores[h, i, j] = dot(q[i, h, :], k[j, kv_head(h), :]) / sqrt(head_dim),
// with -inf written wherever key j is in the future of query i. Query row i has
// absolute position `positions[i]` (device int32 [n_query]) and reads its keys
// through page-table row i * k.table_stride, so one launch serves a prefill
// (one table, positions start_pos + i) or a batch of decodes (a table and a
// position per sequence). n_keys is the score row width: the longest row's key count.
//
// GQA: kv_head(h) = h / (n_heads / n_kv_heads).
// q is [n_query, n_heads, head_dim], k holds n_keys rows of [n_kv_heads, head_dim],
// scores is [n_heads, n_query, n_keys].
void launch_attention_qk(const float* q, const KVPages& k, float* scores, int n_query, int n_keys,
                         int n_heads, int n_kv_heads, int head_dim, const std::int32_t* positions,
                         void* stream = nullptr);

// In-place softmax over the last dimension of an [n_rows, row_len] array.
// Entries masked to -inf by attention_qk must come out as exactly 0.
void launch_softmax_rows(float* x, int n_rows, int row_len, void* stream = nullptr);

// out[i, h, :] = sum_j probs[h, i, j] * v[j, kv_head(h), :] over j <= positions[i],
// with the same per-row positions and page tables as attention_qk.
// probs is [n_heads, n_query, n_keys], v holds n_keys rows of [n_kv_heads, head_dim],
// out is [n_query, n_heads, head_dim].
void launch_attention_av(const float* probs, const KVPages& v, float* out, int n_query, int n_keys,
                         int n_heads, int n_kv_heads, int head_dim, const std::int32_t* positions,
                         void* stream = nullptr);

// out[i] = silu(gate[i]) * up[i], where silu(z) = z * sigmoid(z).
void launch_swiglu(const float* gate, const float* up, float* out, std::size_t n,
                   void* stream = nullptr);

// x[i] += y[i], in place.
void launch_residual_add(float* x, const float* y, std::size_t n, void* stream = nullptr);

// *out_index = argmax_i logits[i], ties resolved to the lowest index (matching
// torch.argmax, so the greedy-decode comparison is exact rather than
// approximate).
void launch_argmax(const float* logits, std::int32_t* out_index, int n, void* stream = nullptr);

}  // namespace llm
