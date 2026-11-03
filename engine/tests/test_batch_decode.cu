// Several sequences decoded together must produce the same ids as each decoded alone.
// Slots are out of order and one stays empty, so a row only finds its cache through row_slot.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "llm/model.h"
#include "ref_fixture.h"

namespace {

constexpr int kSteps = 40;
constexpr int kMaxSeqs = 4;
constexpr int kEmptySlot = 1;
const std::vector<int> kSlots = {3, 0, 2};
const std::vector<std::vector<std::int32_t>> kPrompts = {
    {3, 17, 42, 8, 55},
    {9, 1},
    {60, 2, 33, 71, 5, 14, 27, 40, 11},
};

// Lowest index on ties, like the device argmax.
std::int32_t argmax_row(const std::vector<float>& logits, int row, std::size_t vocab) {
  const auto first = logits.begin() + row * vocab;
  return static_cast<std::int32_t>(std::max_element(first, first + vocab) - first);
}

}  // namespace

int main(int argc, char** argv) {
  using namespace llm::testing;
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <ref_model.llmbin>\n", argv[0]);
    return EXIT_FAILURE;
  }

  std::size_t longest = 0;
  for (const auto& p : kPrompts) longest = std::max(longest, p.size());
  llm::Model model = llm::Model::load(argv[1], static_cast<int>(longest) + kSteps, kMaxSeqs);
  const std::size_t vocab = model.config().vocab_size;
  std::vector<std::vector<std::int32_t>> prompts = kPrompts;
  for (auto& p : prompts) {
    for (std::int32_t& id : p) id %= static_cast<std::int32_t>(vocab);
  }
  const int n_seqs = static_cast<int>(prompts.size());

  // Scattered pages, so neighbouring sequences interleave in the pool.
  model.shuffle_cache_pages(5);

  // Alone: the trusted single-sequence path.
  std::vector<std::vector<std::int32_t>> alone;
  for (const auto& p : prompts) alone.push_back(model.generate(p, kSteps));

  // Together: prefill each into its slot, then one decode pass per step for all of them.
  model.reset_cache();
  std::vector<std::vector<std::int32_t>> together(n_seqs);
  std::vector<std::int32_t> next(n_seqs);
  for (int b = 0; b < n_seqs; ++b) {
    model.forward(prompts[b].data(), static_cast<int>(prompts[b].size()), kSlots[b]);
    next[b] = model.argmax_last(static_cast<int>(prompts[b].size()));
    together[b].push_back(next[b]);
  }
  for (int step = 1; step < kSteps; ++step) {
    model.forward_decode(next.data(), kSlots.data(), n_seqs);
    const std::vector<float> logits = model.logits_host(n_seqs);
    for (int b = 0; b < n_seqs; ++b) {
      next[b] = argmax_row(logits, b, vocab);
      together[b].push_back(next[b]);
    }
  }
  LLM_CUDA_CHECK_LAUNCH();

  bool ok = true;
  for (int b = 0; b < n_seqs; ++b) {
    const std::string label = "sequence " + std::to_string(b) + " in slot " +
                              std::to_string(kSlots[b]);
    ok &= check_int_equal(label.c_str(), together[b], alone[b]);
    const int expect_len = static_cast<int>(prompts[b].size()) + kSteps - 1;
    if (model.cache().length(kSlots[b]) != expect_len) {
      std::fprintf(stderr, "FAIL %s cached %d tokens, expected %d\n", label.c_str(),
                   model.cache().length(kSlots[b]), expect_len);
      ok = false;
    }
  }
  if (model.cache().length(kEmptySlot) != 0 || !model.cache().page_ids(kEmptySlot).empty()) {
    std::fprintf(stderr, "FAIL empty slot %d was touched\n", kEmptySlot);
    ok = false;
  }
  // Different prompts must decode differently, or rows mixed up between sequences could pass.
  if (alone[0] == alone[1] || alone[1] == alone[2] || alone[0] == alone[2]) {
    std::fprintf(stderr, "FAIL two prompts decoded identically, so mixing rows was not tested\n");
    ok = false;
  }

  if (!ok) return EXIT_FAILURE;
  std::printf("PASS: %d sequences decoded together for %d steps match each decoded alone\n",
              n_seqs, kSteps);
  return EXIT_SUCCESS;
}
