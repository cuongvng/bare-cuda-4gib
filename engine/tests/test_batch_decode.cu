// Several sequences decoded together must produce the same ids as each decoded alone.
// Slots are out of order and one stays empty, so a row only finds its cache through row_slot.
// Then generate_batch queues more prompts than fit, under a slot limit and a page limit.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <set>
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

const std::vector<std::vector<std::int32_t>> kQueuePrompts = {
    {3, 17, 42, 8, 55}, {9, 1}, {60, 2, 33, 71, 5, 14, 27, 40, 11}, {44, 7, 19}, {12, 30, 8, 2},
};

struct QueueCase {
  const char* name;
  int max_seqs;
  int kv_pages;   // 0: room for every slot
  int max_live;   // sequences decoding at once, at most and at some point
};

// Each prompt here needs one page, so two pages admit two sequences while three slots admit three.
constexpr QueueCase kQueueCases[] = {
    {"page-limited", 3, 2, 2},
    {"slot-limited", 3, 0, 3},
};

// Lowest index on ties, like the device argmax.
std::int32_t argmax_row(const std::vector<float>& logits, int row, std::size_t vocab) {
  const auto first = logits.begin() + row * vocab;
  return static_cast<std::int32_t>(std::max_element(first, first + vocab) - first);
}

bool check_generate_batch(const char* path, const QueueCase& c, int max_tokens,
                          const std::vector<std::vector<std::int32_t>>& prompts,
                          const std::vector<std::vector<std::int32_t>>& alone) {
  llm::Model model = llm::Model::load(path, max_tokens, c.max_seqs, c.kv_pages);
  model.shuffle_cache_pages(9);

  // Live from a sequence's first token to its last; started records admission order.
  std::set<int> live;
  std::vector<int> started;
  int max_live = 0;
  const auto results = model.generate_batch(prompts, kSteps, [&](int seq, int step, std::int32_t) {
    if (step == 0) {
      live.insert(seq);
      started.push_back(seq);
      max_live = std::max(max_live, static_cast<int>(live.size()));
    }
    if (step == kSteps - 1) live.erase(seq);
  });
  LLM_CUDA_CHECK_LAUNCH();

  bool ok = true;
  for (std::size_t p = 0; p < prompts.size(); ++p) {
    const std::string label = std::string(c.name) + " prompt " + std::to_string(p);
    ok &= llm::testing::check_int_equal(label.c_str(), results[p], alone[p]);
  }
  if (max_live != c.max_live) {
    std::fprintf(stderr, "FAIL %s: %d sequences decoded at once, expected %d\n", c.name, max_live,
                 c.max_live);
    ok = false;
  }
  if (!std::is_sorted(started.begin(), started.end()) || started.size() != prompts.size()) {
    std::fprintf(stderr, "FAIL %s: prompts were not admitted in order\n", c.name);
    ok = false;
  }
  if (model.cache().pool().free_pages() != model.cache().pool().num_pages()) {
    std::fprintf(stderr, "FAIL %s: %d pages still held after every sequence finished\n", c.name,
                 model.cache().pool().pages_in_use());
    ok = false;
  }
  std::printf("%s: %zu prompts, at most %d decoding at once over a %d-page pool\n", c.name,
              prompts.size(), max_live, model.cache().pool().num_pages());
  return ok;
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
    next = model.argmax_rows(n_seqs);
    // The device's per-row winners must match a host argmax over the same logits.
    const std::vector<float> logits = model.logits_host(n_seqs);
    for (int b = 0; b < n_seqs; ++b) {
      if (next[b] != argmax_row(logits, b, vocab)) {
        std::fprintf(stderr, "FAIL step %d row %d: device argmax %d, host argmax %d\n", step, b,
                     next[b], argmax_row(logits, b, vocab));
        return EXIT_FAILURE;
      }
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

  std::vector<std::vector<std::int32_t>> queue_prompts = kQueuePrompts;
  std::vector<std::vector<std::int32_t>> queue_alone;
  for (auto& p : queue_prompts) {
    for (std::int32_t& id : p) id %= static_cast<std::int32_t>(vocab);
    queue_alone.push_back(model.generate(p, kSteps));
  }
  for (const QueueCase& c : kQueueCases) {
    ok &= check_generate_batch(argv[1], c, model.max_tokens(), queue_prompts, queue_alone);
  }

  if (!ok) return EXIT_FAILURE;
  std::printf("PASS: %d sequences decoded together for %d steps match each decoded alone, and "
              "generate_batch matches under slot and page limits\n",
              n_seqs, kSteps);
  return EXIT_SUCCESS;
}
