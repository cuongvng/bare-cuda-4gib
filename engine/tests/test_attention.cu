// Causal GQA attention, checked one stage at a time: raw masked scores, then
// the softmax over them, then the value-weighted output. Checking only the
// final tensor would let a masking bug hide behind a softmax that renormalizes
// it away.
#include <cstdio>
#include <cstdlib>
#include <numeric>

#include "llm/kernels.h"
#include "ref_fixture.h"

int main(int argc, char** argv) {
  using namespace llm::testing;
  const llm::ModelFile fixture = llm::ModelFile::load(fixture_path(argc, argv, "ref_attention.llmbin"));

  const llm::TensorView& q_view = fixture.require("q");
  const llm::TensorView& k_view = fixture.require("k");
  const int n_tokens = static_cast<int>(q_view.dim(0));
  const int n_heads = static_cast<int>(q_view.dim(1));
  const int head_dim = static_cast<int>(q_view.dim(2));
  const int n_kv_heads = static_cast<int>(k_view.dim(1));

  const std::vector<float> scores_expected = load_fp32(fixture, "scores_expected");
  const std::vector<float> probs_expected = load_fp32(fixture, "probs_expected");
  const std::vector<float> out_expected = load_fp32(fixture, "expected");

  llm::DeviceBuffer d_q = upload_fp32(fixture, "q");
  llm::DeviceBuffer d_k = upload_fp32(fixture, "k");
  llm::DeviceBuffer d_v = upload_fp32(fixture, "v");
  llm::DeviceBuffer d_scores(scores_expected.size() * sizeof(float));
  llm::DeviceBuffer d_out(out_expected.size() * sizeof(float));
  d_scores.zero();
  d_out.zero();

  llm::DeviceBuffer d_table = llm::make_device_buffer(std::vector<std::int32_t>{0});
  llm::KVPages k_pages;
  k_pages.pool = d_k.as<float>();
  k_pages.page_table = d_table.as<std::int32_t>();
  k_pages.token_shift = llm::kSinglePageShift;
  llm::KVPages v_pages = k_pages;
  v_pages.pool = d_v.as<float>();

  // A full prefill: query i sits at position i.
  std::vector<std::int32_t> positions(n_tokens);
  std::iota(positions.begin(), positions.end(), 0);
  llm::DeviceBuffer d_positions = llm::make_device_buffer(positions);

  llm::launch_attention_qk(d_q.as<float>(), k_pages, d_scores.as<float>(), n_tokens, n_tokens,
                           n_heads, n_kv_heads, head_dim, d_positions.as<std::int32_t>());
  LLM_CUDA_CHECK_LAUNCH();
  bool ok = check_close("attention_qk", download_fp32(d_scores, scores_expected.size()),
                        scores_expected, 1e-4f, 1e-4f);

  llm::launch_softmax_rows(d_scores.as<float>(), n_heads * n_tokens, n_tokens);
  LLM_CUDA_CHECK_LAUNCH();
  ok &= check_close("softmax", download_fp32(d_scores, probs_expected.size()), probs_expected,
                    1e-6f, 1e-5f);

  llm::launch_attention_av(d_scores.as<float>(), v_pages, d_out.as<float>(), n_tokens, n_tokens,
                           n_heads, n_kv_heads, head_dim, d_positions.as<std::int32_t>());
  LLM_CUDA_CHECK_LAUNCH();
  ok &= check_close("attention_av", download_fp32(d_out, out_expected.size()), out_expected, 1e-5f,
                    1e-5f);

  if (!ok) return EXIT_FAILURE;
  std::printf("PASS: attention matches reference (%d tokens, %d q heads / %d kv heads, dim %d)\n",
              n_tokens, n_heads, n_kv_heads, head_dim);
  return EXIT_SUCCESS;
}
