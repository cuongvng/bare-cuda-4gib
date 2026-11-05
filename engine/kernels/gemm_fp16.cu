#include <cuda_runtime.h>

#include "gemm_tiled.cuh"
#include "kernels.cuh"
#include "llm/kernels.h"

namespace llm {
namespace {

// The prefill shape: M = seq_len rows share one weight matrix, so each weight staged in
// shared memory serves every row of the tile (see gemm_tiled.cuh for the blocking).
struct Fp16Loader {
  const __half* __restrict__ weight;
  int k_in;

  __device__ __forceinline__ float operator()(int n, int k) const {
    return load_half(weight + static_cast<std::size_t>(n) * k_in + k);
  }
};

}  // namespace

void launch_gemm_fp16(const __half* weight, const float* x, float* y, int m_rows, int n_out,
                      int k_in, void* stream) {
  gemm_tiled::launch(Fp16Loader{weight, k_in}, x, y, m_rows, n_out, k_in, stream);
}

}  // namespace llm
