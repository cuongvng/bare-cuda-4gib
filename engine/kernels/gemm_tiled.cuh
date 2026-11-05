#pragma once

// Register-blocked GEMM shared by every weight format: y[m, n] = sum_k x[m, k] * W[n, k].
//
// A block computes a kBlockM x kBlockN patch of y with kThreadsM x kThreadsN threads; each thread
// owns a kRegM x kRegN patch held in registers. Per k step a thread reads kRegM values of x and
// kRegN of W from shared memory and does kRegM * kRegN multiply-adds with them (8 + 8 reads for
// 64 FMAs), instead of 2 reads per FMA when each thread owns one output.
//
// Thread (tx, ty) owns rows ty + i * kThreadsM and columns tx + j * kThreadsN. The interleaving
// keeps a warp's shared-memory reads on consecutive addresses (no bank conflicts) and its writes
// to y coalesced.
//
// The weight format is a Loader: `float operator()(int n, int k) const` returns W[n, k] dequantized.
// It is only called for in-range n and k.

#include <cuda_runtime.h>

#include "kernels.cuh"
#include "llm/cuda_check.h"

namespace llm {
namespace gemm_tiled {

constexpr int kRegM = 8;
constexpr int kRegN = 8;
constexpr int kThreadsM = 8;
constexpr int kThreadsN = 8;
constexpr int kThreads = kThreadsM * kThreadsN;
constexpr int kBlockM = kRegM * kThreadsM;  // 64
constexpr int kBlockN = kRegN * kThreadsN;  // 64
constexpr int kBlockK = 16;
// Tiles are stored k-major ([k][m]); the +1 pad spreads the transposing stores over all banks.
constexpr int kPad = 1;

template <typename Loader>
__global__ void __launch_bounds__(kThreads)
    kernel(Loader load_w, const float* __restrict__ x, float* __restrict__ y, int m_rows,
           int n_out, int k_in) {
  __shared__ float xs[kBlockK][kBlockM + kPad];
  __shared__ float ws[kBlockK][kBlockN + kPad];

  const int tx = threadIdx.x;  // column group
  const int ty = threadIdx.y;  // row group
  const int tid = ty * kThreadsN + tx;
  const int m0 = blockIdx.y * kBlockM;
  const int n0 = blockIdx.x * kBlockN;

  float acc[kRegM][kRegN] = {};

  for (int k0 = 0; k0 < k_in; k0 += kBlockK) {
    // Stage both tiles; consecutive threads take consecutive k, so global reads are coalesced.
    for (int idx = tid; idx < kBlockM * kBlockK; idx += kThreads) {
      const int r = idx / kBlockK;
      const int c = idx % kBlockK;
      const int m = m0 + r;
      const int k = k0 + c;
      xs[c][r] = (m < m_rows && k < k_in) ? x[static_cast<std::size_t>(m) * k_in + k] : 0.0f;
    }
    for (int idx = tid; idx < kBlockN * kBlockK; idx += kThreads) {
      const int r = idx / kBlockK;
      const int c = idx % kBlockK;
      const int n = n0 + r;
      const int k = k0 + c;
      ws[c][r] = (n < n_out && k < k_in) ? load_w(n, k) : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int t = 0; t < kBlockK; ++t) {
      float a[kRegM];
      float b[kRegN];
#pragma unroll
      for (int i = 0; i < kRegM; ++i) a[i] = xs[t][ty + i * kThreadsM];
#pragma unroll
      for (int j = 0; j < kRegN; ++j) b[j] = ws[t][tx + j * kThreadsN];
#pragma unroll
      for (int i = 0; i < kRegM; ++i) {
#pragma unroll
        for (int j = 0; j < kRegN; ++j) acc[i][j] += a[i] * b[j];
      }
    }
    __syncthreads();  // before the next chunk overwrites the tiles
  }

#pragma unroll
  for (int i = 0; i < kRegM; ++i) {
    const int m = m0 + ty + i * kThreadsM;
    if (m >= m_rows) continue;
#pragma unroll
    for (int j = 0; j < kRegN; ++j) {
      const int n = n0 + tx + j * kThreadsN;
      if (n < n_out) y[static_cast<std::size_t>(m) * n_out + n] = acc[i][j];
    }
  }
}

template <typename Loader>
void launch(const Loader& load_w, const float* x, float* y, int m_rows, int n_out, int k_in,
            void* stream) {
  if (m_rows == 0 || n_out == 0 || k_in == 0) return;
  const dim3 block(kThreadsN, kThreadsM);
  const dim3 grid(ceil_div(n_out, kBlockN), ceil_div(m_rows, kBlockM));
  kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(load_w, x, y, m_rows, n_out, k_in);
  LLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace gemm_tiled
}  // namespace llm
