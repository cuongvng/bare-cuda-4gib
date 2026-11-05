#pragma once

// Shared plumbing for the reference tests. Each test loads a fixture written by
// tools/references/reference_forward.py -- itself a .llmbin file, so the loader
// proved in the previous milestone is the only container code involved -- pulls
// named tensors out of it, and compares kernel output against the torch values.
//
// No test framework, by the project's dependency policy: a test is a main()
// that returns non-zero.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "llm/cuda_check.h"
#include "llm/device_buffer.h"
#include "llm/loader.h"

namespace llm {
namespace testing {

inline std::string fixture_path(int argc, char** argv, const char* what) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <%s>\n", argv[0], what);
    std::exit(EXIT_FAILURE);
  }
  return argv[1];
}

inline std::size_t elem_count(const TensorView& t) {
  return t.nbytes / dtype_size_bytes(t.dtype);
}

// Widen a fixture tensor to FP32 regardless of how it was stored, so a test
// never has to care whether the Python side wrote a weight or an activation.
inline std::vector<float> load_fp32(const ModelFile& file, const std::string& name) {
  const TensorView& t = file.require(name);
  const std::size_t n = elem_count(t);
  std::vector<float> out(n);
  if (t.dtype == DType::kFP32) {
    std::memcpy(out.data(), t.data, n * sizeof(float));
  } else if (t.dtype == DType::kFP16) {
    const auto* src = reinterpret_cast<const __half*>(t.data);
    for (std::size_t i = 0; i < n; ++i) out[i] = __half2float(src[i]);
  } else {
    std::fprintf(stderr, "fixture tensor '%s' has an unsupported dtype\n", name.c_str());
    std::exit(EXIT_FAILURE);
  }
  return out;
}

// Integer payloads (token ids, positions, argmax results) ride in FP32 tensors:
// the binary format's dtype tag has no int32. Values are small and exact.
inline std::vector<std::int32_t> load_int32(const ModelFile& file, const std::string& name) {
  const std::vector<float> f = load_fp32(file, name);
  std::vector<std::int32_t> out(f.size());
  for (std::size_t i = 0; i < f.size(); ++i) out[i] = static_cast<std::int32_t>(std::lround(f[i]));
  return out;
}

// Upload an FP16 tensor to VRAM exactly as stored -- no widening, since that is
// what the kernels read.
inline DeviceBuffer upload_half(const ModelFile& file, const std::string& name) {
  const TensorView& t = file.require(name);
  if (t.dtype != DType::kFP16) {
    std::fprintf(stderr, "fixture tensor '%s' is not FP16\n", name.c_str());
    std::exit(EXIT_FAILURE);
  }
  DeviceBuffer buf(t.nbytes);
  buf.upload(t.data, t.nbytes);
  return buf;
}

// Upload a tensor's bytes as-is, whatever its dtype (e.g. packed int8/int4 weights).
inline DeviceBuffer upload_raw(const ModelFile& file, const std::string& name) {
  const TensorView& t = file.require(name);
  DeviceBuffer buf(t.nbytes);
  buf.upload(t.data, t.nbytes);
  return buf;
}

inline DeviceBuffer upload_fp32(const ModelFile& file, const std::string& name) {
  const std::vector<float> host = load_fp32(file, name);
  return make_device_buffer(host);
}

// Repeats a [rows, width] row-major matrix cyclically to want_rows rows, for shapes beyond a fixture.
inline std::vector<float> repeat_rows(const std::vector<float>& v, int rows, int width, int want_rows) {
  std::vector<float> out(static_cast<std::size_t>(want_rows) * width);
  for (int r = 0; r < want_rows; ++r) {
    std::copy_n(v.begin() + static_cast<std::size_t>(r % rows) * width, width,
                out.begin() + static_cast<std::size_t>(r) * width);
  }
  return out;
}

inline std::vector<float> download_fp32(const DeviceBuffer& buf, std::size_t count) {
  std::vector<float> out(count);
  buf.download(out.data(), count * sizeof(float));
  return out;
}

// Tolerance shape: absolute for values near zero, relative for large ones. The
// GPU accumulates in a different order than torch does, so bitwise equality is
// not on offer for anything with a reduction in it -- but the error must stay
// at rounding scale, not drift with the reduction length.
inline bool check_close(const char* label, const std::vector<float>& got,
                        const std::vector<float>& expected, float atol, float rtol) {
  if (got.size() != expected.size()) {
    std::fprintf(stderr, "FAIL %s: size %zu, expected %zu\n", label, got.size(), expected.size());
    return false;
  }
  std::size_t bad = 0;
  float worst = 0.0f;
  std::size_t worst_i = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const float g = got[i], e = expected[i];
    // Masked score entries are exactly -inf on both sides; treat matching
    // infinities as equal instead of as a NaN-producing difference.
    if (std::isinf(g) && std::isinf(e) && ((g > 0) == (e > 0))) continue;
    const float diff = std::fabs(g - e);
    const float limit = atol + rtol * std::fabs(e);
    if (!(diff <= limit)) {
      if (bad < 5) {
        std::fprintf(stderr, "  %s[%zu]: got %g, expected %g (diff %g > %g)\n", label, i, g, e,
                     diff, limit);
      }
      ++bad;
    }
    if (diff > worst) {
      worst = diff;
      worst_i = i;
    }
  }
  if (bad != 0) {
    std::fprintf(stderr, "FAIL %s: %zu / %zu elements outside tolerance (worst %g at %zu)\n", label,
                 bad, got.size(), worst, worst_i);
    return false;
  }
  std::printf("  ok %-18s %zu elements, worst abs diff %g\n", label, got.size(), worst);
  return true;
}

inline bool check_int_equal(const char* label, const std::vector<std::int32_t>& got,
                            const std::vector<std::int32_t>& expected) {
  if (got.size() != expected.size()) {
    std::fprintf(stderr, "FAIL %s: size %zu, expected %zu\n", label, got.size(), expected.size());
    return false;
  }
  std::size_t bad = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (got[i] != expected[i]) {
      if (bad < 5) {
        std::fprintf(stderr, "  %s[%zu]: got %d, expected %d\n", label, i, got[i], expected[i]);
      }
      ++bad;
    }
  }
  if (bad != 0) {
    std::fprintf(stderr, "FAIL %s: %zu / %zu ids differ\n", label, bad, got.size());
    return false;
  }
  std::printf("  ok %-18s %zu ids match exactly\n", label, got.size());
  return true;
}

}  // namespace testing
}  // namespace llm
