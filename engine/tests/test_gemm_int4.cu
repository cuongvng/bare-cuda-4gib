// INT4-weight GEMM vs torch on the dequantized weights, at two group sizes.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "llm/kernels.h"
#include "ref_fixture.h"

int main(int argc, char** argv) {
  using namespace llm::testing;
  const llm::ModelFile fixture =
      llm::ModelFile::load(fixture_path(argc, argv, "ref_gemm_int4.llmbin"));

  for (const std::string p : {"g32.", "g128."}) {
    const llm::TensorView& w_view = fixture.require(p + "weight");
    const int n_out = static_cast<int>(w_view.dim(0));
    const int k_in = static_cast<int>(w_view.dim(1));
    const int m_rows = static_cast<int>(fixture.require(p + "input").dim(0));
    const int group = k_in / static_cast<int>(fixture.require(p + "scales").dim(1));
    const std::vector<float> expected = load_fp32(fixture, p + "expected");

    llm::DeviceBuffer d_w = upload_raw(fixture, p + "weight");
    llm::DeviceBuffer d_s = upload_half(fixture, p + "scales");
    llm::DeviceBuffer d_m = upload_half(fixture, p + "mins");
    llm::DeviceBuffer d_x = upload_fp32(fixture, p + "input");
    llm::DeviceBuffer d_y(expected.size() * sizeof(float));
    d_y.zero();

    llm::launch_gemm_int4(d_w.as<std::uint8_t>(), d_s.as<__half>(), d_m.as<__half>(),
                          d_x.as<float>(), d_y.as<float>(), m_rows, n_out, k_in, group);
    LLM_CUDA_CHECK_LAUNCH();

    const std::vector<float> got = download_fp32(d_y, expected.size());
    const std::string label = "gemm_int4 " + p;
    if (!check_close(label.c_str(), got, expected, 2e-3f, 2e-3f)) return EXIT_FAILURE;

    // M = 1 through the GEMM must agree with row 0 of the full result.
    llm::DeviceBuffer d_y1(static_cast<std::size_t>(n_out) * sizeof(float));
    d_y1.zero();
    llm::launch_gemm_int4(d_w.as<std::uint8_t>(), d_s.as<__half>(), d_m.as<__half>(),
                          d_x.as<float>(), d_y1.as<float>(), 1, n_out, k_in, group);
    LLM_CUDA_CHECK_LAUNCH();
    const std::vector<float> got_row0 = download_fp32(d_y1, n_out);
    const std::vector<float> expected_row0(expected.begin(), expected.begin() + n_out);
    const std::string label1 = label + "(M=1)";
    if (!check_close(label1.c_str(), got_row0, expected_row0, 2e-3f, 2e-3f)) return EXIT_FAILURE;

    std::printf("  %s: %d x %d x %d, group %d\n", label.c_str(), m_rows, n_out, k_in, group);
  }

  std::printf("PASS: gemm_int4 matches reference\n");
  return EXIT_SUCCESS;
}
