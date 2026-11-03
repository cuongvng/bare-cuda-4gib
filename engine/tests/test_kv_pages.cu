// Attention over shuffled pages must equal attention over one unbroken buffer,
// which test_attention already ties to the torch reference.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

Result run_attention(const float* q, const llm::KVPages& k, const llm::KVPages& v, int n_keys,
                     const std::vector<std::int32_t>& positions) {
  const int n_query = static_cast<int>(positions.size());
  Result result;
  result.scores.resize(static_cast<std::size_t>(kHeads) * n_query * n_keys);
  result.out.resize(static_cast<std::size_t>(n_query) * kHeads * kHeadDim);
  llm::DeviceBuffer d_scores(result.scores.size() * sizeof(float));
  llm::DeviceBuffer d_out(result.out.size() * sizeof(float));
  llm::DeviceBuffer d_positions = llm::make_device_buffer(positions);
  const std::int32_t* pos = d_positions.as<std::int32_t>();

  llm::launch_attention_qk(q, k, d_scores.as<float>(), n_query, n_keys, kHeads, kKvHeads, kHeadDim,
                           pos);
  llm::launch_softmax_rows(d_scores.as<float>(), kHeads * n_query, n_keys);
  llm::launch_attention_av(d_scores.as<float>(), v, d_out.as<float>(), n_query, n_keys, kHeads,
                           kKvHeads, kHeadDim, pos);
  LLM_CUDA_CHECK(cudaDeviceSynchronize());

  d_scores.download(result.scores.data(), result.scores.size() * sizeof(float));
  d_out.download(result.out.data(), result.out.size() * sizeof(float));
  return result;
}

Result run_case(const llm::DeviceBuffer& q, const llm::KVPages& k, const llm::KVPages& v,
                const Case& c) {
  std::vector<std::int32_t> positions(c.n_query);
  for (int i = 0; i < c.n_query; ++i) positions[i] = c.pos_offset + i;
  return run_attention(q.as<float>(), k, v, c.n_keys, positions);
}

// A sequence of the batched case: `first_page` indexes the shuffled table, its rows start there.
struct BatchSeq {
  int first_page;
  int length;
};

constexpr BatchSeq kBatch[] = {{0, 13}, {4, 4}, {5, 9}};
constexpr int kBatchSize = sizeof(kBatch) / sizeof(kBatch[0]);
constexpr int kBatchTablePages = 4;  // the longest sequence's page count

// Several decodes in one launch, each through its own table row, must equal each run alone.
// Table rows are stored in reverse, so batch row b only finds its pages through row_slot.
bool check_batched_decode(const std::vector<float>& q_rows, const llm::DeviceBuffer& d_q,
                          const llm::KVPages& k_paged, const llm::KVPages& v_paged,
                          const std::vector<std::int32_t>& table, std::int32_t padding_page) {
  std::vector<std::int32_t> batch_table(kBatchSize * kBatchTablePages, padding_page);
  std::vector<std::int32_t> row_slot;
  std::vector<float> batch_q;
  std::vector<std::int32_t> positions;
  int n_keys = 0;
  for (int b = 0; b < kBatchSize; ++b) {
    const BatchSeq& s = kBatch[b];
    const int pages = (s.length + kTokensPerPage - 1) / kTokensPerPage;
    row_slot.push_back(kBatchSize - 1 - b);
    std::copy_n(table.begin() + s.first_page, pages,
                batch_table.begin() + row_slot.back() * kBatchTablePages);
    const std::size_t q_row = static_cast<std::size_t>(s.first_page * kTokensPerPage + s.length - 1);
    batch_q.insert(batch_q.end(), q_rows.begin() + q_row * kHeads * kHeadDim,
                   q_rows.begin() + (q_row + 1) * kHeads * kHeadDim);
    positions.push_back(s.length - 1);
    n_keys = std::max(n_keys, s.length);
  }

  llm::DeviceBuffer d_batch_table = llm::make_device_buffer(batch_table);
  llm::DeviceBuffer d_batch_q = llm::make_device_buffer(batch_q);
  llm::DeviceBuffer d_row_slot = llm::make_device_buffer(row_slot);
  llm::KVPages k_batch = k_paged;
  k_batch.page_table = d_batch_table.as<std::int32_t>();
  k_batch.table_stride = kBatchTablePages;
  k_batch.row_slot = d_row_slot.as<std::int32_t>();
  llm::KVPages v_batch = v_paged;
  v_batch.page_table = k_batch.page_table;
  v_batch.table_stride = kBatchTablePages;
  v_batch.row_slot = k_batch.row_slot;
  const Result got = run_attention(d_batch_q.as<float>(), k_batch, v_batch, n_keys, positions);

  bool ok = true;
  const std::size_t out_row = static_cast<std::size_t>(kHeads) * kHeadDim;
  for (int b = 0; b < kBatchSize; ++b) {
    const BatchSeq& s = kBatch[b];
    llm::KVPages k_alone = k_paged;
    k_alone.page_table += s.first_page;
    llm::KVPages v_alone = v_paged;
    v_alone.page_table += s.first_page;
    const std::size_t q_row = static_cast<std::size_t>(s.first_page * kTokensPerPage + s.length - 1);
    const Result alone = run_attention(d_q.as<float>() + q_row * out_row, k_alone, v_alone,
                                       s.length, {s.length - 1});

    if (!std::equal(alone.out.begin(), alone.out.end(), got.out.begin() + b * out_row)) {
      std::fprintf(stderr, "FAIL batched decode: sequence %d output differs from running alone\n",
                   b);
      ok = false;
    }
    // Probabilities: equal up to the sequence's length, exactly 0 past it.
    for (int h = 0; h < kHeads; ++h) {
      const float* batch_row =
          got.scores.data() + (static_cast<std::size_t>(h) * kBatchSize + b) * n_keys;
      const float* alone_row = alone.scores.data() + static_cast<std::size_t>(h) * s.length;
      const bool same = std::equal(alone_row, alone_row + s.length, batch_row) &&
                        std::all_of(batch_row + s.length, batch_row + n_keys,
                                    [](float p) { return p == 0.0f; });
      if (!same) {
        std::fprintf(stderr, "FAIL batched decode: sequence %d head %d probabilities differ\n", b,
                     h);
        ok = false;
      }
    }
  }
  return ok;
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
  pool.allocate(kPageFloats * sizeof(float), kNumPages + 1);
  std::vector<std::int32_t> table(kNumPages);
  for (std::int32_t& page : table) page = pool.acquire();
  std::shuffle(table.begin(), table.end(), rng);
  // Unused table entries point here; NaN survives a multiply by 0, so any read past a sequence shows.
  const std::int32_t padding_page = pool.acquire();

  const std::size_t k_offset = kTestLayer * kLayerFloats;
  const std::size_t v_offset = (kNumLayers + kTestLayer) * kLayerFloats;
  std::vector<float> pool_host(kPageFloats * (kNumPages + 1), kPoison);
  std::fill_n(pool_host.begin() + padding_page * kPageFloats, kPageFloats,
              std::numeric_limits<float>::quiet_NaN());
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
    const Result expected = run_case(d_q, k_single, v_single, c);
    const Result got = run_case(d_q, k_paged, v_paged, c);
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
  if (run_case(d_q, k_swapped, v_swapped, first).out ==
      run_case(d_q, k_single, v_single, first).out) {
    std::fprintf(stderr, "FAIL a wrong page table went unnoticed\n");
    ok = false;
  }

  ok &= check_batched_decode(q_rows, d_q, k_paged, v_paged, table, padding_page);

  if (!ok) return EXIT_FAILURE;
  std::printf("PASS: shuffled pages match an unbroken buffer in %zu cases, and %d batched decodes "
              "match running alone\n",
              sizeof(kCases) / sizeof(kCases[0]), kBatchSize);
  return EXIT_SUCCESS;
}
