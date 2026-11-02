#pragma once

// Shared device-side helpers for the hand-written kernels: launch-config
// arithmetic, half->float reads, and the warp/block reductions that most of
// the bandwidth-bound kernels are built out of.
//
// Everything here is header-only and __forceinline__ on purpose. These are the
// innermost few instructions of every kernel; a real call would cost more than
// the work it wraps, and separable compilation is off (engine/CMakeLists.txt)
// so cross-translation-unit device calls aren't available anyway.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>

#include "llm/kernels.h"

namespace llm {

constexpr int kWarpSize = 32;

// Default block size for the elementwise/bandwidth-bound kernels: 256 threads
// is 8 warps, enough to hide global-memory latency on sm_86 without pinning so
// many registers that occupancy drops.
constexpr int kDefaultBlockSize = 256;

// A block that reduces one row: 1024 caps at the hardware maximum, and the
// reduction kernels are written to loop when the row is longer than the block.
constexpr int kReduceBlockSize = 256;

// sm_86 allows 65535 blocks in the x dimension; grid-stride kernels clamp to
// this rather than growing the grid with the problem size.
constexpr int kMaxGridSize = 65535;

constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

inline int clamped_grid(int want) {
  if (want < 1) return 1;
  return want > kMaxGridSize ? kMaxGridSize : want;
}

// Start of this layer's rows in the page holding `token`; costs one page table read.
__device__ __forceinline__ const float* kv_page_rows(const KVPages& pages, int token) {
  const std::size_t page = pages.page_table[token >> pages.token_shift];
  return pages.pool + page * pages.page_stride + pages.layer_offset;
}

__device__ __forceinline__ int kv_page_row(const KVPages& pages, int token) {
  return token & ((1 << pages.token_shift) - 1);
}

// Weights are read straight out of their FP16 VRAM layout and widened in
// registers -- the whole point of storing them narrow is that the widening
// never touches memory.
__device__ __forceinline__ float load_half(const __half* p) { return __half2float(*p); }

// Sum `value` across the 32 lanes of one warp; every lane gets the total.
//
// The exchange is a butterfly (__shfl_xor_sync) rather than the more common
// __shfl_down_sync, precisely so the result lands in every lane instead of only
// lane 0 -- the callers below all need the reduced value back to scale their own
// elements with. It costs the same five steps either way, and none of it touches
// memory: the whole reduction lives in the register file.
__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

// Max of `value` across one warp. Same butterfly as the sum.
__device__ __forceinline__ float warp_reduce_max(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  return value;
}

// Sum `value` across a whole block. `shared` must be at least
// (blockDim.x / kWarpSize) floats. The result is valid in *every* thread.
//
// Two stages: the warps reduce themselves through shuffles, each warp's lane 0
// writes one float to shared memory, and then every thread reads those few
// partials back. Only warp-count floats ever cross shared memory -- eight of
// them for a 256-thread block -- so the serial tail is cheaper than a second
// shuffle round plus the broadcast it would need.
//
// The trailing __syncthreads() is what makes repeated calls on the same `shared`
// buffer safe: without it the next call's writes could land while a straggling
// thread is still reading this call's partials.
__device__ __forceinline__ float block_reduce_sum(float value, float* shared) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int warps = blockDim.x / kWarpSize;

  value = warp_reduce_sum(value);
  if (lane == 0) shared[warp] = value;
  __syncthreads();

  float total = 0.0f;
  for (int i = 0; i < warps; ++i) total += shared[i];
  __syncthreads();
  return total;
}

// Max across a whole block; same two-stage structure as block_reduce_sum, with
// -FLT_MAX as the identity for threads that hold no value.
__device__ __forceinline__ float block_reduce_max(float value, float* shared) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int warps = blockDim.x / kWarpSize;

  value = warp_reduce_max(value);
  if (lane == 0) shared[warp] = value;
  __syncthreads();

  float total = -FLT_MAX;
  for (int i = 0; i < warps; ++i) total = fmaxf(total, shared[i]);
  __syncthreads();
  return total;
}

}  // namespace llm
