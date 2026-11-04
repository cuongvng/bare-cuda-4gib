#include <cuda_runtime.h>

#include <cstdint>

#include "kernels.cuh"
#include "llm/cuda_check.h"
#include "llm/kernels.h"

namespace llm {
namespace {

// One warp per output row; each lane reads one 32-bit word = 8 packed weights, so a warp step covers 256.
constexpr int kWarpsPerBlock = kReduceBlockSize / kWarpSize;
constexpr int kPerLane = 8;
constexpr int kPerWarpStep = kWarpSize * kPerLane;

__global__ void gemv_int4_kernel(const std::uint8_t* __restrict__ weight,
                                 const __half* __restrict__ scales, const __half* __restrict__ mins,
                                 const float* __restrict__ x, float* __restrict__ y, int n_out,
                                 int k_in, int group_size) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= n_out) return;

  const int groups = k_in / group_size;
  const auto* w_row =
      reinterpret_cast<const std::uint32_t*>(weight + static_cast<std::size_t>(row) * (k_in / 2));
  const std::size_t g_row = static_cast<std::size_t>(row) * groups;

  float acc = 0.0f;
  for (int k = lane * kPerLane; k < k_in; k += kPerWarpStep) {
    const std::uint32_t packed = w_row[k / kPerLane];
    // x[k .. k+7] in two 16-byte loads; k is a multiple of 8, so both are aligned.
    const float4 lo = *reinterpret_cast<const float4*>(x + k);
    const float4 hi = *reinterpret_cast<const float4*>(x + k + 4);
    const float xs[kPerLane] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    // Nibble j of the little-endian word is weight k + j (even k in each byte's low nibble).
    float qx = 0.0f, sum_x = 0.0f;
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
      qx += static_cast<float>((packed >> (4 * j)) & 0xFu) * xs[j];
      sum_x += xs[j];
    }
    // sum((q*s + m) * x) = s * sum(q*x) + m * sum(x); all 8 values share one group.
    const int g = k / group_size;
    acc += load_half(scales + g_row + g) * qx + load_half(mins + g_row + g) * sum_x;
  }

  acc = warp_reduce_sum(acc);
  if (lane == 0) y[row] = acc;
}

// Same per-warp layout, but each packed word is unpacked once and applied to up to kMaxRows inputs.
template <int kMaxRows>
__global__ void gemv_int4_batch_kernel(const std::uint8_t* __restrict__ weight,
                                       const __half* __restrict__ scales,
                                       const __half* __restrict__ mins, const float* __restrict__ x,
                                       float* __restrict__ y, int m_rows, int n_out, int k_in,
                                       int group_size) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= n_out) return;

  const int groups = k_in / group_size;
  const auto* w_row =
      reinterpret_cast<const std::uint32_t*>(weight + static_cast<std::size_t>(row) * (k_in / 2));
  const std::size_t g_row = static_cast<std::size_t>(row) * groups;

  float acc[kMaxRows] = {};
  for (int k = lane * kPerLane; k < k_in; k += kPerWarpStep) {
    const std::uint32_t packed = w_row[k / kPerLane];
    float q[kPerLane];
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) q[j] = static_cast<float>((packed >> (4 * j)) & 0xFu);
    const int g = k / group_size;
    const float scale = load_half(scales + g_row + g);
    const float min = load_half(mins + g_row + g);

#pragma unroll
    for (int r = 0; r < kMaxRows; ++r) {
      if (r >= m_rows) break;
      const float* xr = x + static_cast<std::size_t>(r) * k_in + k;
      const float4 lo = *reinterpret_cast<const float4*>(xr);
      const float4 hi = *reinterpret_cast<const float4*>(xr + 4);
      const float xs[kPerLane] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
      float qx = 0.0f, sum_x = 0.0f;
#pragma unroll
      for (int j = 0; j < kPerLane; ++j) {
        qx += q[j] * xs[j];
        sum_x += xs[j];
      }
      acc[r] += scale * qx + min * sum_x;
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

void launch_gemv_int4(const std::uint8_t* weight, const __half* scales, const __half* mins,
                      const float* x, float* y, int n_out, int k_in, int group_size, void* stream) {
  if (n_out == 0 || k_in == 0) return;
  // The kernel reads x as float4.
  if (reinterpret_cast<std::uintptr_t>(x) % alignof(float4) != 0) {
    LLM_CUDA_CHECK(cudaErrorMisalignedAddress);
  }
  const int grid = ceil_div(n_out, kWarpsPerBlock);
  gemv_int4_kernel<<<grid, kReduceBlockSize, 0, static_cast<cudaStream_t>(stream)>>>(
      weight, scales, mins, x, y, n_out, k_in, group_size);
  LLM_CUDA_CHECK(cudaGetLastError());
}

void launch_gemv_int4_batch(const std::uint8_t* weight, const __half* scales, const __half* mins,
                            const float* x, float* y, int m_rows, int n_out, int k_in,
                            int group_size, void* stream) {
  if (m_rows == 0 || n_out == 0 || k_in == 0) return;
  if (m_rows > kGemvBatchMaxRows) LLM_CUDA_CHECK(cudaErrorInvalidValue);
  if (reinterpret_cast<std::uintptr_t>(x) % alignof(float4) != 0) {
    LLM_CUDA_CHECK(cudaErrorMisalignedAddress);
  }
  const int grid = ceil_div(n_out, kWarpsPerBlock);
  const auto s = static_cast<cudaStream_t>(stream);
  // Smallest instance that fits, so few rows don't pay for 16 accumulators.
  if (m_rows <= 4) {
    gemv_int4_batch_kernel<4><<<grid, kReduceBlockSize, 0, s>>>(weight, scales, mins, x, y, m_rows,
                                                                n_out, k_in, group_size);
  } else if (m_rows <= 8) {
    gemv_int4_batch_kernel<8><<<grid, kReduceBlockSize, 0, s>>>(weight, scales, mins, x, y, m_rows,
                                                                n_out, k_in, group_size);
  } else {
    gemv_int4_batch_kernel<kGemvBatchMaxRows><<<grid, kReduceBlockSize, 0, s>>>(
        weight, scales, mins, x, y, m_rows, n_out, k_in, group_size);
  }
  LLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace llm
