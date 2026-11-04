#include <cuda_runtime.h>

#include <cstdint>

#include "kernels.cuh"
#include "llm/cuda_check.h"
#include "llm/kernels.h"

namespace llm {
namespace {

// Same shape as gemv_fp16 (one warp per output row), but each lane reads 4 int8
// weights as one 32-bit word, so a warp step pulls 128 bytes = 128 weights.
constexpr int kWarpsPerBlock = kReduceBlockSize / kWarpSize;
constexpr int kPerLane = 4;
constexpr int kPerWarpStep = kWarpSize * kPerLane;

__global__ void gemv_int8_kernel(const std::int8_t* __restrict__ weight,
                                 const __half* __restrict__ scales, const float* __restrict__ x,
                                 float* __restrict__ y, int n_out, int k_in, int group_size) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= n_out) return;

  const int groups = k_in / group_size;
  const auto* w_row = reinterpret_cast<const std::uint32_t*>(weight + static_cast<std::size_t>(row) * k_in);
  const __half* s_row = scales + static_cast<std::size_t>(row) * groups;

  float acc = 0.0f;
  for (int k = lane * kPerLane; k < k_in; k += kPerWarpStep) {
    const std::uint32_t packed = w_row[k / kPerLane];
    // The 4 values share one group because group_size is a multiple of 4.
    const float dot = static_cast<float>(static_cast<std::int8_t>(packed)) * x[k] +
                      static_cast<float>(static_cast<std::int8_t>(packed >> 8)) * x[k + 1] +
                      static_cast<float>(static_cast<std::int8_t>(packed >> 16)) * x[k + 2] +
                      static_cast<float>(static_cast<std::int8_t>(packed >> 24)) * x[k + 3];
    acc += load_half(s_row + k / group_size) * dot;
  }

  acc = warp_reduce_sum(acc);
  if (lane == 0) y[row] = acc;
}

// Same per-warp layout, but each packed word is unpacked once and applied to up to kMaxRows inputs.
template <int kMaxRows>
__global__ void gemv_int8_batch_kernel(const std::int8_t* __restrict__ weight,
                                       const __half* __restrict__ scales, const float* __restrict__ x,
                                       float* __restrict__ y, int m_rows, int n_out, int k_in,
                                       int group_size) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= n_out) return;

  const int groups = k_in / group_size;
  const auto* w_row = reinterpret_cast<const std::uint32_t*>(weight + static_cast<std::size_t>(row) * k_in);
  const __half* s_row = scales + static_cast<std::size_t>(row) * groups;

  float acc[kMaxRows] = {};
  for (int k = lane * kPerLane; k < k_in; k += kPerWarpStep) {
    const std::uint32_t packed = w_row[k / kPerLane];
    const float q0 = static_cast<float>(static_cast<std::int8_t>(packed));
    const float q1 = static_cast<float>(static_cast<std::int8_t>(packed >> 8));
    const float q2 = static_cast<float>(static_cast<std::int8_t>(packed >> 16));
    const float q3 = static_cast<float>(static_cast<std::int8_t>(packed >> 24));
    const float scale = load_half(s_row + k / group_size);

#pragma unroll
    for (int r = 0; r < kMaxRows; ++r) {
      if (r >= m_rows) break;
      const float* xr = x + static_cast<std::size_t>(r) * k_in + k;
      acc[r] += scale * (q0 * xr[0] + q1 * xr[1] + q2 * xr[2] + q3 * xr[3]);
    }
  }

#pragma unroll
  for (int r = 0; r < kMaxRows; ++r) {
    if (r >= m_rows) break;
    const float sum = warp_reduce_sum(acc[r]);
    if (lane == 0) y[static_cast<std::size_t>(r) * n_out + row] = sum;
  }
}

}  // namespace

void launch_gemv_int8(const std::int8_t* weight, const __half* scales, const float* x, float* y,
                      int n_out, int k_in, int group_size, void* stream) {
  if (n_out == 0 || k_in == 0) return;
  const int grid = ceil_div(n_out, kWarpsPerBlock);
  gemv_int8_kernel<<<grid, kReduceBlockSize, 0, static_cast<cudaStream_t>(stream)>>>(
      weight, scales, x, y, n_out, k_in, group_size);
  LLM_CUDA_CHECK(cudaGetLastError());
}

void launch_gemv_int8_batch(const std::int8_t* weight, const __half* scales, const float* x,
                            float* y, int m_rows, int n_out, int k_in, int group_size,
                            void* stream) {
  if (m_rows == 0 || n_out == 0 || k_in == 0) return;
  if (m_rows > kGemvBatchMaxRows) LLM_CUDA_CHECK(cudaErrorInvalidValue);
  const int grid = ceil_div(n_out, kWarpsPerBlock);
  const auto s = static_cast<cudaStream_t>(stream);
  // Smallest instance that fits, so few rows don't pay for 16 accumulators.
  if (m_rows <= 4) {
    gemv_int8_batch_kernel<4><<<grid, kReduceBlockSize, 0, s>>>(weight, scales, x, y, m_rows, n_out,
                                                                k_in, group_size);
  } else if (m_rows <= 8) {
    gemv_int8_batch_kernel<8><<<grid, kReduceBlockSize, 0, s>>>(weight, scales, x, y, m_rows, n_out,
                                                                k_in, group_size);
  } else {
    gemv_int8_batch_kernel<kGemvBatchMaxRows><<<grid, kReduceBlockSize, 0, s>>>(
        weight, scales, x, y, m_rows, n_out, k_in, group_size);
  }
  LLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace llm
