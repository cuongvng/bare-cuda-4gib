#include <cuda_runtime.h>

#include "kernels.cuh"
#include "llm/cuda_check.h"
#include "llm/kernels.h"

namespace llm {
namespace {

// probs * V. Mirror image of attention_qk: the probability row is reused across
// every column of head_dim, and the V reads -- the other half of the KV cache
// traffic -- are the bandwidth cost that matters.
//
// One block per (head, query), one thread per output column, accumulating
// across keys. Each thread walks V down the key axis; consecutive threads read
// consecutive columns of the same V row, so the access is coalesced even though
// the loop is over keys.
__global__ void attention_av_kernel(const float* __restrict__ probs, const KVPages v_all,
                                    float* __restrict__ out, int n_query, int n_keys, int n_heads,
                                    int n_kv_heads, int head_dim,
                                    const std::int32_t* __restrict__ positions) {
  const int query = blockIdx.x;
  const int head = blockIdx.y;
  const int kv_head = head / (n_heads / n_kv_heads);
  const KVPages v = kv_pages_for_row(v_all, query);
  // Keys past this row's position may lie beyond its sequence's pages, so they are never read.
  const int key_end = positions[query] + 1;

  const float* prob_row = probs + (static_cast<std::size_t>(head) * n_query + query) * n_keys;
  float* out_row = out + (static_cast<std::size_t>(query) * n_heads + head) * head_dim;

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    // Consecutive threads hold consecutive d, so each step of the loop is one
    // coalesced read of a V row; prob_row[j] is the same address for the whole
    // block and broadcasts.
    float acc = 0.0f;
    const float* page_rows = nullptr;
    for (int key = 0; key < key_end; ++key) {
      const int row = kv_page_row(v, key);
      // Keys walk in order, so row 0 is the only place a new page can begin.
      if (row == 0) page_rows = kv_page_rows(v, key);
      acc += prob_row[key] * page_rows[(row * n_kv_heads + kv_head) * head_dim + d];
    }
    out_row[d] = acc;
  }
}

}  // namespace

void launch_attention_av(const float* probs, const KVPages& v, float* out, int n_query, int n_keys,
                         int n_heads, int n_kv_heads, int head_dim, const std::int32_t* positions,
                         void* stream) {
  if (n_query == 0 || n_keys == 0 || n_heads == 0) return;
  const dim3 grid(n_query, n_heads);
  const int block = head_dim <= kDefaultBlockSize ? head_dim : kDefaultBlockSize;
  attention_av_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      probs, v, out, n_query, n_keys, n_heads, n_kv_heads, head_dim, positions);
  LLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace llm
