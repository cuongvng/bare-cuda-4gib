// Attention over shuffled pages must equal attention over one unbroken buffer,
// which test_attention already ties to the torch reference.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "llm/allocator.h"
#include "llm/kernels.h"

namespace {

constexpr int kTokenShift = 2;
constexpr int kTokensPerPage = 1 << kTokenShift;
constexpr int kNumLayers = 2;
constexpr int kTestLayer = 1;
constexpr int kNumPages = 8;
constexpr int kMaxTokens = kNumPages * kTokensPerPage;
constexpr int kHeads = 4;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 8;
constexpr int kKvDim = kKvHeads * kHeadDim;
constexpr std::size_t kLayerFloats = kTokensPerPage * kKvDim;
constexpr std::size_t kPageFloats = 2 * kNumLayers * kLayerFloats;  // K then V, for every layer
constexpr float kPoison = 1e6f;  // fills every float the kernels must never read

struct Case {
  const char* name;
  int n_query;
  int n_keys;
  int pos_offset;
};

constexpr Case kCases[] = {
    {"prefill across pages", 23, 23, 0},
    {"prefill ending on a page boundary", 8, 8, 0},
    {"prefill inside one page", 3, 3, 0},
    {"chunk after a prefill", 4, 12, 8},
    {"decode, last row of a page", 1, 20, 19},
    {"decode, first row of a page", 1, 21, 20},
    {"decode, all pages full", 1, kMaxTokens, kMaxTokens - 1},
};

std::vector<float> random_floats(std::size_t count, std::mt19937& rng) {
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> values(count);
  for (float& value : values) value = normal(rng);
  return values;
}

struct Result {
  std::vector<float> scores;
  std::vector<float> out;
};

Result run_attention(const llm::DeviceBuffer& q, const llm::KVPages& k, const llm::KVPages& v,
                     const Case& c) {
  Result result;
  result.scores.resize(static_cast<std::size_t>(kHeads) * c.n_query * c.n_keys);
  result.out.resize(static_cast<std::size_t>(c.n_query) * kHeads * kHeadDim);
  llm::DeviceBuffer d_scores(result.scores.size() * sizeof(float));
  llm::DeviceBuffer d_out(result.out.size() * sizeof(float));

  llm::launch_attention_qk(q.as<float>(), k, d_scores.as<float>(), c.n_query, c.n_keys, kHeads,
                           kKvHeads, kHeadDim, c.pos_offset);
  llm::launch_softmax_rows(d_scores.as<float>(), kHeads * c.n_query, c.n_keys);
  llm::launch_attention_av(d_scores.as<float>(), v, d_out.as<float>(), c.n_query, c.n_keys, kHeads,
                           kKvHeads, kHeadDim);
  LLM_CUDA_CHECK(cudaDeviceSynchronize());

  d_scores.download(result.scores.data(), result.scores.size() * sizeof(float));
  d_out.download(result.out.data(), result.out.size() * sizeof(float));
  return result;
}

}  // namespace

int main() {
  std::mt19937 rng(7);
  const std::size_t rows_floats = static_cast<std::size_t>(kMaxTokens) * kKvDim;
  const std::vector<float> k_rows = random_floats(rows_floats, rng);
  const std::vector<float> v_rows = random_floats(rows_floats, rng);
  const std::vector<float> q_rows =
      random_floats(static_cast<std::size_t>(kMaxTokens) * kHeads * kHeadDim, rng);

  llm::DeviceBuffer d_q = llm::make_device_buffer(q_rows);
  llm::DeviceBuffer d_k = llm::make_device_buffer(k_rows);
  llm::DeviceBuffer d_v = llm::make_device_buffer(v_rows);

  // The trusted path: one unbroken buffer.
  llm::DeviceBuffer d_single = llm::make_device_buffer(std::vector<std::int32_t>{0});
  llm::KVPages k_single;
  k_single.pool = d_k.as<float>();
  k_single.page_table = d_single.as<std::int32_t>();
  k_single.token_shift = llm::kSinglePageShift;
  llm::KVPages v_single = k_single;
  v_single.pool = d_v.as<float>();

  // The path under test: the same rows copied into shuffled pages of a real pool.
  llm::PagePool pool;
  pool.allocate(kPageFloats * sizeof(float), kNumPages);
  std::vector<std::int32_t> table(kNumPages);
  for (std::int32_t& page : table) page = pool.acquire();
  std::shuffle(table.begin(), table.end(), rng);

  const std::size_t k_offset = kTestLayer * kLayerFloats;
  const std::size_t v_offset = (kNumLayers + kTestLayer) * kLayerFloats;
  std::vector<float> pool_host(kPageFloats * kNumPages, kPoison);
  for (int i = 0; i < kNumPages; ++i) {
    const std::size_t rows_start = static_cast<std::size_t>(i) * kLayerFloats;
    float* page_host = pool_host.data() + table[i] * kPageFloats;
    std::copy_n(k_rows.begin() + rows_start, kLayerFloats, page_host + k_offset);
    std::copy_n(v_rows.begin() + rows_start, kLayerFloats, page_host + v_offset);
  }
  LLM_CUDA_CHECK(cudaMemcpy(pool.page_data(0), pool_host.data(), pool.total_bytes(),
                            cudaMemcpyHostToDevice));

  llm::DeviceBuffer d_table = llm::make_device_buffer(table);
  llm::KVPages k_paged;
  k_paged.pool = pool.page_data(0);
  k_paged.page_table = d_table.as<std::int32_t>();
  k_paged.page_stride = kPageFloats;
  k_paged.layer_offset = k_offset;
  k_paged.token_shift = kTokenShift;
  llm::KVPages v_paged = k_paged;
  v_paged.layer_offset = v_offset;

  bool ok = true;
  for (const Case& c : kCases) {
    const Result expected = run_attention(d_q, k_single, v_single, c);
    const Result got = run_attention(d_q, k_paged, v_paged, c);
    if (got.scores != expected.scores || got.out != expected.out) {
      std::fprintf(stderr, "FAIL %s: paged result differs\n", c.name);
      ok = false;
    }
  }

  // A deliberately wrong table must be caught, or the comparison above proves nothing.
  std::vector<std::int32_t> swapped = table;
  std::swap(swapped[0], swapped[1]);
  llm::DeviceBuffer d_swapped = llm::make_device_buffer(swapped);
  llm::KVPages k_swapped = k_paged;
  k_swapped.page_table = d_swapped.as<std::int32_t>();
  llm::KVPages v_swapped = v_paged;
  v_swapped.page_table = d_swapped.as<std::int32_t>();
  const Case& first = kCases[0];
  if (run_attention(d_q, k_swapped, v_swapped, first).out ==
      run_attention(d_q, k_single, v_single, first).out) {
    std::fprintf(stderr, "FAIL a wrong page table went unnoticed\n");
    ok = false;
  }

  if (!ok) return EXIT_FAILURE;
  std::printf("PASS: shuffled pages match an unbroken buffer in %zu cases\n",
              sizeof(kCases) / sizeof(kCases[0]));
  return EXIT_SUCCESS;
}
