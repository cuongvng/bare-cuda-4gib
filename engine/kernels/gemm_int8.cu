#include <cuda_runtime.h>

#include <cstdint>

#include "gemm_tiled.cuh"
#include "kernels.cuh"
#include "llm/kernels.h"

namespace llm {
namespace {

// Dequantizes q * scale on the way into shared memory, so the inner loop is format-free.
struct Int8Loader {
  const std::int8_t* __restrict__ weight;
  const __half* __restrict__ scales;
  int k_in;
  int group_size;

  __device__ __forceinline__ float operator()(int n, int k) const {
    const float q = static_cast<float>(weight[static_cast<std::size_t>(n) * k_in + k]);
    return q * load_half(scales + static_cast<std::size_t>(n) * (k_in / group_size) + k / group_size);
  }
};

}  // namespace

void launch_gemm_int8(const std::int8_t* weight, const __half* scales, const float* x, float* y,
                      int m_rows, int n_out, int k_in, int group_size, void* stream) {
  gemm_tiled::launch(Int8Loader{weight, scales, k_in, group_size}, x, y, m_rows, n_out, k_in,
                     stream);
}

}  // namespace llm
