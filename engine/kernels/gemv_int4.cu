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
    // Nibble j of the little-endian word is weight k + j (even k in each byte's low nibble).
    float qx = 0.0f, sum_x = 0.0f;
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
      const float xj = x[k + j];
      qx += static_cast<float>((packed >> (4 * j)) & 0xFu) * xj;
      sum_x += xj;
    }
    // sum((q*s + m) * x) = s * sum(q*x) + m * sum(x); all 8 values share one group.
    const int g = k / group_size;
    acc += load_half(scales + g_row + g) * qx + load_half(mins + g_row + g) * sum_x;
  }

  acc = warp_reduce_sum(acc);
  if (lane == 0) y[row] = acc;
}

}  // namespace

void launch_gemv_int4(const std::uint8_t* weight, const __half* scales, const __half* mins,
                      const float* x, float* y, int n_out, int k_in, int group_size, void* stream) {
  if (n_out == 0 || k_in == 0) return;
  const int grid = ceil_div(n_out, kWarpsPerBlock);
  gemv_int4_kernel<<<grid, kReduceBlockSize, 0, static_cast<cudaStream_t>(stream)>>>(
      weight, scales, mins, x, y, n_out, k_in, group_size);
  LLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace llm
