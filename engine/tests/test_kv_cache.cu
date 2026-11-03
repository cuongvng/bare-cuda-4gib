// Checks the cached decode path against the cacheless one on the same model:
// incremental decode, then one full pass over the sequence it produced. A
// cache, position or masking bug shows up here without routing through torch.
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "llm/model.h"
#include "ref_fixture.h"

namespace {

constexpr int kSteps = 200;
constexpr std::uint32_t kShuffleSeed = 2;  // gives pages 2 0 3 1
constexpr std::size_t kMiB = 1 << 20;
constexpr std::size_t kVramTargetBytes = 256 * kMiB;  // big enough to dwarf allocation rounding
constexpr std::size_t kVramToleranceBytes = 4 * kMiB;

std::size_t free_vram() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  LLM_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  return free_bytes;
}

std::size_t gap(std::size_t a, std::size_t b) { return a > b ? a - b : b - a; }

// The card's own free-memory count must move by what the cache says it holds.
bool check_cache_vram(const llm::ModelConfig& config) {
  const std::size_t token_bytes = 2 * sizeof(float) * config.num_layers * config.num_kv_heads *
                                  config.head_dim;
  const int tokens = static_cast<int>(kVramTargetBytes / token_bytes);

  const std::size_t before = free_vram();
  std::size_t reported = 0;
  std::size_t taken = 0;
  {
    llm::KVCache cache;
    cache.allocate(config, tokens);
    reported = cache.bytes();
    taken = before - free_vram();
  }
  const std::size_t returned = before - free_vram();

  std::printf("vram: cache reports %.1f MiB, card lost %.1f MiB, %.1f MiB still missing after free\n",
              reported / double(kMiB), taken / double(kMiB), returned / double(kMiB));
  bool ok = true;
  if (gap(taken, reported) > kVramToleranceBytes) {
    std::fprintf(stderr, "FAIL cache size differs from what the card lost\n");
    ok = false;
  }
  if (returned > kVramToleranceBytes) {
    std::fprintf(stderr, "FAIL cache memory not returned to the card\n");
    ok = false;
  }
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace llm::testing;
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <ref_model.llmbin>\n", argv[0]);
    return EXIT_FAILURE;
  }

  std::vector<std::int32_t> prompt = {3, 17, 42, 8, 55};
  const int max_tokens = static_cast<int>(prompt.size()) + kSteps;
  llm::Model model = llm::Model::load(argv[1], max_tokens);
  for (std::int32_t& id : prompt) id %= static_cast<std::int32_t>(model.config().vocab_size);

  // Scattered pages, so tokens next to each other are not next to each other in memory.
  model.shuffle_cache_pages(kShuffleSeed);

  // Incremental: prefill once, then one token per step out of the cache.
  const std::vector<std::int32_t> incremental = model.generate(prompt, kSteps);
  LLM_CUDA_CHECK_LAUNCH();

  const std::vector<std::int32_t>& pages = model.cache().page_ids();
  std::printf("pages used, in token order:");
  for (const std::int32_t page : pages) std::printf(" %d", page);
  std::printf("\n");
  const auto in_order = [](std::int32_t a, std::int32_t b) { return b == a + 1; };
  if (pages.size() < 2 || std::adjacent_find(pages.begin(), pages.end(), in_order) != pages.end()) {
    std::fprintf(stderr, "FAIL some neighbouring pages are in order, so scattering was not tested\n");
    return EXIT_FAILURE;
  }

  std::vector<std::int32_t> sequence = prompt;
  sequence.insert(sequence.end(), incremental.begin(), incremental.end());

  // generate() stops before forwarding its last token; this extends the cache
  // over the whole sequence so the two logits rows describe the same thing.
  model.forward(&sequence.back(), 1);
  const std::vector<float> cached_row = model.logits_host(1);

  // Cacheless: re-run the whole prefix every step, which is what the cache
  // claims to be an optimization of.
  std::vector<std::int32_t> cacheless;
  std::vector<std::int32_t> ids = prompt;
  for (int step = 0; step < kSteps; ++step) {
    model.reset_cache();
    model.forward(ids.data(), static_cast<int>(ids.size()));
    const std::int32_t next = model.argmax_last(static_cast<int>(ids.size()));
    cacheless.push_back(next);
    ids.push_back(next);
  }

  model.reset_cache();
  model.forward(sequence.data(), static_cast<int>(sequence.size()));
  const std::vector<float> full = model.logits_host(static_cast<int>(sequence.size()));
  const std::size_t vocab = model.config().vocab_size;
  const std::vector<float> cacheless_row(full.end() - vocab, full.end());

  std::printf("model: %u layers, hidden %u, vocab %u -- cache %.1f KiB over %d tokens\n",
              model.config().num_layers, model.config().hidden_size, model.config().vocab_size,
              model.cache_bytes() / 1024.0, max_tokens);

  // Ids must match exactly; logits get FP32-rounding slack since decode (GEMV) and prefill (GEMM) sum in different orders.
  bool ok = check_int_equal("decoded ids", incremental, cacheless);
  ok &= check_close("final logits", cached_row, cacheless_row, 1e-5f, 1e-5f);
  ok &= check_cache_vram(model.config());

  if (!ok) return EXIT_FAILURE;
  std::printf("PASS: %d incremental decode steps over scattered pages match a cacheless pass exactly\n",
              kSteps);
  return EXIT_SUCCESS;
}
