#include <cuda_runtime.h>

#include <cstdint>

#include "gemm_tiled.cuh"
#include "kernels.cuh"
#include "llm/kernels.h"

namespace llm {
namespace {

// Unpacks one nibble (low nibble = even k) and applies q * scale + min on the way into shared memory.
struct Int4Loader {
  const std::uint8_t* __restrict__ weight;
  const __half* __restrict__ scales;
  const __half* __restrict__ mins;
  int k_in;
  int group_size;

  __device__ __forceinline__ float operator()(int n, int k) const {
    const std::uint8_t b = weight[static_cast<std::size_t>(n) * (k_in / 2) + k / 2];
    const float q = static_cast<float>((k & 1) ? (b >> 4) : (b & 0xF));
    const std::size_t g = static_cast<std::size_t>(n) * (k_in / group_size) + k / group_size;
    return q * load_half(scales + g) + load_half(mins + g);
  }
};

}  // namespace

void launch_gemm_int4(const std::uint8_t* weight, const __half* scales, const __half* mins,
                      const float* x, float* y, int m_rows, int n_out, int k_in, int group_size,
                      void* stream) {
  gemm_tiled::launch(Int4Loader{weight, scales, mins, k_in, group_size}, x, y, m_rows, n_out, k_in,
                     stream);
}

}  // namespace llm
