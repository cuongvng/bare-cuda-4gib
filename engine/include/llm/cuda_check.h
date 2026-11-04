#pragma once

#include <cuda_runtime.h>

#include <string>

#include "llm/error.h"

namespace llm {

inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
  if (err != cudaSuccess) {
    raise_error(std::string("[CUDA] ") + file + ":" + std::to_string(line) + ": " + expr +
                " failed: " + cudaGetErrorName(err) + " (" + cudaGetErrorString(err) + ")");
  }
}

}  // namespace llm

// Wrap every runtime call. Fails loudly at the call site instead of surfacing
// as a confusing error from an unrelated later call.
#define LLM_CUDA_CHECK(expr) ::llm::cuda_check((expr), #expr, __FILE__, __LINE__)

// Launches report errors asynchronously; this catches both the launch-config
// error (cudaGetLastError) and any fault during execution (the sync).
#define LLM_CUDA_CHECK_LAUNCH()                  \
  do {                                           \
    LLM_CUDA_CHECK(cudaGetLastError());          \
    LLM_CUDA_CHECK(cudaDeviceSynchronize());     \
  } while (0)
