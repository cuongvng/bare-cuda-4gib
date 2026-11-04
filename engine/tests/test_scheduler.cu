// The scheduler must give each request the ids it would get decoded alone, however requests
// arrive, end (budget, stop id) or get cancelled, and hand every page back at the end.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "llm/model.h"
#include "llm/scheduler.h"
#include "ref_fixture.h"

namespace {

constexpr int kSteps = 24;
constexpr int kMaxSeqs = 3;
const std::vector<std::vector<std::int32_t>> kPrompts = {
    {3, 17, 42, 8, 55}, {9, 1}, {60, 2, 33, 71, 5, 14, 27, 40, 11}, {44, 7, 19},
};

struct Outcome {
  std::map<std::int64_t, std::vector<std::int32_t>> tokens;
  std::map<std::int64_t, llm::FinishReason> finish;
  bool ok = true;
};

// Folds events into per-request results, checking indices count up and only the last one finishes.
void collect(const std::vector<llm::TokenEvent>& events, Outcome& out) {
  for (const llm::TokenEvent& e : events) {
    if (out.finish.count(e.id)) {
      std::fprintf(stderr, "FAIL request %lld got an event after finishing\n",
                   static_cast<long long>(e.id));
      out.ok = false;
    }
    auto& toks = out.tokens[e.id];
    if (e.index != static_cast<int>(toks.size())) {
      std::fprintf(stderr, "FAIL request %lld event index %d, expected %zu\n",
                   static_cast<long long>(e.id), e.index, toks.size());
      out.ok = false;
    }
    toks.push_back(e.token);
    if (e.finish != llm::FinishReason::kNone) out.finish[e.id] = e.finish;
  }
}

void run_to_idle(llm::Scheduler& s, Outcome& out) {
  while (!s.idle()) collect(s.step(), out);
}

llm::Request make(std::int64_t id, const std::vector<std::int32_t>& prompt, int max_new,
                  std::vector<std::int32_t> stop_ids = {}) {
  llm::Request r;
  r.id = id;
  r.prompt = prompt;
  r.max_new_tokens = max_new;
  r.stop_ids = std::move(stop_ids);
  return r;
}

llm::Request sampled(std::int64_t id, const std::vector<std::int32_t>& prompt, float top_p,
                     std::int64_t seed) {
  llm::Request r = make(id, prompt, kSteps);
  r.sampling.temperature = 1.0f;
  r.sampling.top_p = top_p;
  r.sampling.seed = seed;
  return r;
}

bool expect(const char* label, const Outcome& out, std::int64_t id,
            const std::vector<std::int32_t>& tokens, llm::FinishReason finish) {
  const auto t = out.tokens.find(id);
  const std::vector<std::int32_t> got = t == out.tokens.end() ? std::vector<std::int32_t>{} : t->second;
  const std::string name = std::string(label) + " request " + std::to_string(id);
  bool ok = llm::testing::check_int_equal(name.c_str(), got, tokens);
  const auto f = out.finish.find(id);
  const llm::FinishReason got_finish = f == out.finish.end() ? llm::FinishReason::kNone : f->second;
  if (got_finish != finish) {
    std::fprintf(stderr, "FAIL %s finished with '%s', expected '%s'\n", name.c_str(),
                 llm::finish_reason_name(got_finish), llm::finish_reason_name(finish));
    ok = false;
  }
  return ok;
}

bool all_pages_free(const char* label, const llm::Model& model) {
  if (model.cache().pool().free_pages() == model.cache().pool().num_pages()) return true;
  std::fprintf(stderr, "FAIL %s: %d pages still held\n", label, model.cache().pool().pages_in_use());
  return false;
}

std::vector<std::int32_t> prefix(const std::vector<std::int32_t>& v, std::size_t n) {
  return std::vector<std::int32_t>(v.begin(), v.begin() + std::min(n, v.size()));
}

}  // namespace

int main(int argc, char** argv) {
  using llm::FinishReason;
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <ref_model.llmbin>\n", argv[0]);
    return EXIT_FAILURE;
  }

  std::size_t longest = 0;
  for (const auto& p : kPrompts) longest = std::max(longest, p.size());
  llm::Model model = llm::Model::load(argv[1], static_cast<int>(longest) + kSteps, kMaxSeqs);
  std::vector<std::vector<std::int32_t>> prompts = kPrompts;
  for (auto& p : prompts) {
    for (std::int32_t& id : p) id %= static_cast<std::int32_t>(model.config().vocab_size);
  }
  const int n = static_cast<int>(prompts.size());

  std::vector<std::vector<std::int32_t>> alone;
  for (const auto& p : prompts) alone.push_back(model.generate(p, kSteps));
  model.shuffle_cache_pages(7);
  bool ok = true;

  // Late arrivals: two requests start, two more join mid-run, one more than there are slots.
  {
    llm::Scheduler s(model);
    Outcome out;
    s.submit(make(0, prompts[0], kSteps));
    s.submit(make(1, prompts[1], kSteps));
    for (int i = 0; i < 5; ++i) collect(s.step(), out);
    s.submit(make(2, prompts[2], kSteps));
    s.submit(make(3, prompts[3], kSteps));
    run_to_idle(s, out);
    for (int p = 0; p < n; ++p) ok &= expect("late", out, p, alone[p], FinishReason::kLength);
    ok &= out.ok && all_pages_free("late", model);
  }

  // Own budgets and stop ids: a short budget, a stop id, and a request done at prefill.
  {
    const std::int32_t stop = alone[1][kSteps / 2];
    const auto first = std::find(alone[1].begin(), alone[1].end(), stop) - alone[1].begin();
    llm::Scheduler s(model);
    Outcome out;
    s.submit(make(0, prompts[0], 7));
    s.submit(make(1, prompts[1], kSteps, {stop}));
    s.submit(make(2, prompts[2], 1));
    run_to_idle(s, out);
    ok &= expect("budget", out, 0, prefix(alone[0], 7), FinishReason::kLength);
    ok &= expect("stop", out, 1, prefix(alone[1], first + 1), FinishReason::kStop);
    ok &= expect("prefill-only", out, 2, prefix(alone[2], 1), FinishReason::kLength);
    ok &= out.ok && all_pages_free("budget/stop", model);
  }

  // Cancel one running and one still waiting; the others carry on untouched.
  {
    llm::Scheduler s(model);
    Outcome out;
    for (int p = 0; p < n; ++p) s.submit(make(p, prompts[p], kSteps));
    for (int i = 0; i < 3; ++i) collect(s.step(), out);
    if (s.num_running() != kMaxSeqs || s.num_waiting() != 1) {
      std::fprintf(stderr, "FAIL cancel setup: %d running, %d waiting\n", s.num_running(),
                   s.num_waiting());
      ok = false;
    }
    const std::size_t before = out.tokens[1].size();
    ok &= s.cancel(1) && s.cancel(3) && !s.cancel(99);
    run_to_idle(s, out);
    ok &= expect("cancel", out, 0, alone[0], FinishReason::kLength);
    ok &= expect("cancel", out, 1, prefix(alone[1], before), FinishReason::kNone);
    ok &= expect("cancel", out, 2, alone[2], FinishReason::kLength);
    ok &= expect("cancel", out, 3, {}, FinishReason::kNone);
    ok &= out.ok && all_pages_free("cancel", model);
  }

  // Sampling: a tiny top-p keeps only the top token, so it must equal greedy through the host path;
  // a seeded draw must repeat exactly whether decoded alone or batched with others.
  {
    llm::Scheduler s(model);
    Outcome out;
    s.submit(sampled(0, prompts[0], 1e-6f, -1));
    s.submit(sampled(1, prompts[1], 1.0f, 7));
    run_to_idle(s, out);
    const std::vector<std::int32_t> seeded_alone = out.tokens[1];

    llm::Scheduler batched(model);
    Outcome out2;
    batched.submit(make(0, prompts[0], kSteps));
    batched.submit(sampled(1, prompts[1], 1.0f, 7));
    batched.submit(sampled(2, prompts[1], 1.0f, 8));
    for (int i = 0; i < 3; ++i) collect(batched.step(), out2);
    batched.submit(make(3, prompts[3], kSteps));
    run_to_idle(batched, out2);

    ok &= expect("top_p~0", out, 0, alone[0], FinishReason::kLength);
    ok &= expect("seed 7 batched", out2, 1, seeded_alone, FinishReason::kLength);
    ok &= expect("greedy beside sampled", out2, 0, alone[0], FinishReason::kLength);
    ok &= expect("greedy beside sampled", out2, 3, alone[3], FinishReason::kLength);
    if (seeded_alone == alone[1] || out2.tokens[2] == seeded_alone) {
      std::fprintf(stderr, "FAIL sampling at T=1 matched greedy, or seeds 7 and 8 drew the same\n");
      ok = false;
    }
    ok &= out.ok && out2.ok && all_pages_free("sampling", model);
  }

  // check() refuses what could never run.
  {
    llm::Scheduler s(model);
    const llm::EngineLimits& lim = s.limits();
    const std::vector<std::int32_t> too_long(lim.max_tokens + 1, 1);
    if (llm::Scheduler::check(lim, {}, 4).empty() || llm::Scheduler::check(lim, prompts[0], 0).empty() ||
        llm::Scheduler::check(lim, too_long, 4).empty() ||
        !llm::Scheduler::check(lim, prompts[0], 1 << 20).empty()) {
      std::fprintf(stderr, "FAIL check() accepted a bad request or refused a good one\n");
      ok = false;
    }
    if (llm::Scheduler::check(lim, sampled(0, prompts[0], 0.0f, 1)).empty()) {
      std::fprintf(stderr, "FAIL check() accepted top_p = 0\n");
      ok = false;
    }
  }

  if (!ok) return EXIT_FAILURE;
  std::printf("PASS: late arrivals, budgets, stop ids, cancels and sampling all behave\n");
  return EXIT_SUCCESS;
}
