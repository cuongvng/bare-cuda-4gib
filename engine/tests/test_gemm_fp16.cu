// FP16-weight GEMM (the prefill-shape matmul) and batched GEMV vs the torch reference.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "llm/kernels.h"
#include "ref_fixture.h"

int main(int argc, char** argv) {
  using namespace llm::testing;
  const llm::ModelFile fixture = llm::ModelFile::load(fixture_path(argc, argv, "ref_gemm.llmbin"));

  const llm::TensorView& w_view = fixture.require("weight");
  const llm::TensorView& x_view = fixture.require("input");
  const int n_out = static_cast<int>(w_view.dim(0));
  const int k_in = static_cast<int>(w_view.dim(1));
  const int m_rows = static_cast<int>(x_view.dim(0));
  const std::vector<float> expected = load_fp32(fixture, "expected");

  llm::DeviceBuffer d_w = upload_half(fixture, "weight");
  llm::DeviceBuffer d_x = upload_fp32(fixture, "input");
  llm::DeviceBuffer d_y(expected.size() * sizeof(float));
  d_y.zero();

  llm::launch_gemm_fp16(d_w.as<__half>(), d_x.as<float>(), d_y.as<float>(), m_rows, n_out, k_in);
  LLM_CUDA_CHECK_LAUNCH();

  const std::vector<float> got = download_fp32(d_y, expected.size());
  if (!check_close("gemm", got, expected, 2e-3f, 2e-3f)) return EXIT_FAILURE;

  // M and K are both ragged against the tile size in the fixture, so the
  // partial-tile path is already covered above. This second call pins the
  // single-row case: prefill and decode must go through the same kernel and
  // agree, since decode is just M = 1.
  llm::DeviceBuffer d_y1(static_cast<std::size_t>(n_out) * sizeof(float));
  d_y1.zero();
  llm::launch_gemm_fp16(d_w.as<__half>(), d_x.as<float>(), d_y1.as<float>(), 1, n_out, k_in);
  LLM_CUDA_CHECK_LAUNCH();
  const std::vector<float> got_row0 = download_fp32(d_y1, n_out);
  const std::vector<float> expected_row0(expected.begin(), expected.begin() + n_out);
  if (!check_close("gemm(M=1)", got_row0, expected_row0, 2e-3f, 2e-3f)) return EXIT_FAILURE;

  // 150 rows (the fixture's rows repeated) span several row tiles, the last one ragged.
  {
    constexpr int kBigM = 150;
    const std::vector<float> x_big = repeat_rows(load_fp32(fixture, "input"), m_rows, k_in, kBigM);
    const llm::DeviceBuffer d_xb = llm::make_device_buffer(x_big);
    llm::DeviceBuffer d_yb(static_cast<std::size_t>(kBigM) * n_out * sizeof(float));
    llm::launch_gemm_fp16(d_w.as<__half>(), d_xb.as<float>(), d_yb.as<float>(), kBigM, n_out, k_in);
    LLM_CUDA_CHECK_LAUNCH();
    const std::vector<float> got = download_fp32(d_yb, static_cast<std::size_t>(kBigM) * n_out);
    const std::vector<float> want = repeat_rows(expected, m_rows, n_out, kBigM);
    const std::string big_label = std::string("gemm(M=150)");
    if (!check_close(big_label.c_str(), got, want, 2e-3f, 2e-3f)) return EXIT_FAILURE;
  }

  // Batched GEMV on the first m rows, at sizes on each side of the kernel's instance boundaries.
  for (const int m : {2, 3, 4, 5, 8, 9, 16}) {
    llm::DeviceBuffer d_yb(static_cast<std::size_t>(m) * n_out * sizeof(float));
    d_yb.zero();
    llm::launch_gemv_fp16_batch(d_w.as<__half>(), d_x.as<float>(), d_yb.as<float>(), m, n_out,
                                k_in);
    LLM_CUDA_CHECK_LAUNCH();
    const std::vector<float> got_b = download_fp32(d_yb, static_cast<std::size_t>(m) * n_out);
    const std::vector<float> expected_b(expected.begin(), expected.begin() + m * n_out);
    const std::string label_b = "gemv_fp16_batch(M=" + std::to_string(m) + ")";
    if (!check_close(label_b.c_str(), got_b, expected_b, 2e-3f, 2e-3f)) return EXIT_FAILURE;
  }

  std::printf("PASS: gemm_fp16 and gemv_fp16_batch match reference (%d x %d x %d)\n", m_rows, n_out, k_in);
  return EXIT_SUCCESS;
}
