#include <cuda_runtime.h>

#include "kernels.cuh"
#include "llm/cuda_check.h"
#include "llm/kernels.h"

namespace llm {
namespace {

// Q*K^T with causal masking. The query vector for a given (query, head) is
// reused against every key, so it is staged in shared memory once per block and
// the K reads -- which dominate, and which become the KV-cache reads once the
// cache lands -- stream through coalesced and are never re-read.
//
// One block per (head, query): head_dim floats of shared memory, one warp-shaped
// walk over the keys.
__global__ void attention_qk_kernel(const float* __restrict__ q, const KVPages k_all,
                                    float* __restrict__ scores, int n_query, int n_keys,
                                    int n_heads, int n_kv_heads, int head_dim,
                                    const std::int32_t* __restrict__ positions) {
  extern __shared__ float q_shared[];

  const int query = blockIdx.x;
  const int head = blockIdx.y;
  const int kv_head = head / (n_heads / n_kv_heads);
  const KVPages k = kv_pages_for_row(k_all, query);

  // Staged once, then read by every key below -- the one piece of reuse in this
  // kernel, and the reason it is worth a barrier.
  const float* q_vec = q + (static_cast<std::size_t>(query) * n_heads + head) * head_dim;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) q_shared[d] = q_vec[d];
  __syncthreads();

  const int last_key = positions[query];
  const float scale = rsqrtf(static_cast<float>(head_dim));
  float* score_row = scores + (static_cast<std::size_t>(head) * n_query + query) * n_keys;

  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int warps = blockDim.x / kWarpSize;

  // A warp per key rather than a thread per key: with lanes walking head_dim,
  // the K reads are consecutive addresses within a row. One thread per key would
  // stride by n_kv_heads * head_dim and scatter every request -- and these are
  // the reads that become the KV-cache traffic.
  int current_page = -1;
  const float* page_rows = nullptr;
  for (int key = warp; key < n_keys; key += warps) {
    if (key > last_key) {
      // Masked entries are written, not skipped: softmax_rows reads the whole
      // row, and -INFINITY is what makes it produce exactly 0 there.
      if (lane == 0) score_row[key] = -INFINITY;
      continue;
    }

    // The page table is read only when this warp walks onto a new page.
    const int page = key >> k.token_shift;
    if (page != current_page) {
      page_rows = kv_page_rows(k, key);
      current_page = page;
    }

    const float* k_vec = page_rows + (kv_page_row(k, key) * n_kv_heads + kv_head) * head_dim;
    float acc = 0.0f;
    for (int d = lane; d < head_dim; d += kWarpSize) acc += q_shared[d] * k_vec[d];

    acc = warp_reduce_sum(acc);
    if (lane == 0) score_row[key] = acc * scale;
  }
}

}  // namespace

void launch_attention_qk(const float* q, const KVPages& k, float* scores, int n_query, int n_keys,
                         int n_heads, int n_kv_heads, int head_dim, const std::int32_t* positions,
                         void* stream) {
  if (n_query == 0 || n_keys == 0 || n_heads == 0) return;
  const dim3 grid(n_query, n_heads);
  const std::size_t shared_bytes = head_dim * sizeof(float);
  attention_qk_kernel<<<grid, kDefaultBlockSize, shared_bytes, static_cast<cudaStream_t>(stream)>>>(
      q, k, scores, n_query, n_keys, n_heads, n_kv_heads, head_dim, positions);
  LLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace llm
