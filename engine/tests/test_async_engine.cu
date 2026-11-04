// Requests submitted from several threads at once, while the worker decodes, must each get
// the ids they get decoded alone. Refusals (bad, duplicate, queue full), cancels, a failed
// load and a worker failure must all come back as statuses and events, not a dead process.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "llm/async_engine.h"
#include "llm/error.h"
#include "llm/model.h"
#include "ref_fixture.h"

namespace {

constexpr int kSteps = 24;
constexpr int kMaxSeqs = 3;
constexpr int kThreads = 3;
const std::vector<std::vector<std::int32_t>> kPrompts = {
    {3, 17, 42, 8, 55}, {9, 1}, {60, 2, 33, 71, 5, 14, 27, 40, 11},
    {44, 7, 19},        {12, 30, 8, 2}, {5, 5, 21},
};

llm::Request make(std::int64_t id, const std::vector<std::int32_t>& prompt, int max_new) {
  llm::Request r;
  r.id = id;
  r.prompt = prompt;
  r.max_new_tokens = max_new;
  return r;
}

// Collects events until `count` requests have finished or a minute passes.
struct Drain {
  llm::AsyncEngine& engine;
  std::map<std::int64_t, std::vector<std::int32_t>> got;
  std::map<std::int64_t, llm::FinishReason> finish;

  void until(std::size_t count) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (finish.size() < count && std::chrono::steady_clock::now() < deadline) {
      for (const llm::TokenEvent& e : engine.poll(100)) {
        if (e.token >= 0) got[e.id].push_back(e.token);
        if (e.finish != llm::FinishReason::kNone) finish[e.id] = e.finish;
      }
    }
  }
};

}  // namespace

int main(int argc, char** argv) {
  using llm::FinishReason;
  using llm::SubmitStatus;
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <ref_model.llmbin>\n", argv[0]);
    return EXIT_FAILURE;
  }

  std::size_t longest = 0;
  for (const auto& p : kPrompts) longest = std::max(longest, p.size());
  const int max_tokens = static_cast<int>(longest) + kSteps;

  std::vector<std::vector<std::int32_t>> prompts = kPrompts;
  std::vector<std::vector<std::int32_t>> alone;
  {
    llm::Model model = llm::Model::load(argv[1], max_tokens, 1);
    for (auto& p : prompts) {
      for (std::int32_t& id : p) id %= static_cast<std::int32_t>(model.config().vocab_size);
      alone.push_back(model.generate(p, kSteps));
    }
  }
  const int n = static_cast<int>(prompts.size());
  bool ok = true;

  llm::EngineOptions options;
  options.model_path = argv[1];
  options.max_tokens = max_tokens;
  options.max_seqs = kMaxSeqs;

  // A model that cannot load throws from the constructor instead of exiting.
  {
    llm::EngineOptions bad = options;
    bad.model_path = "/nonexistent/model.llmbin";
    bool threw = false;
    try {
      llm::AsyncEngine broken(bad);
    } catch (const llm::Error&) {
      threw = true;
    }
    if (!threw) {
      std::fprintf(stderr, "FAIL loading a missing model did not throw\n");
      ok = false;
    }
  }

  llm::AsyncEngine engine(options);
  Drain drain{engine};

  // Refusals come back at once, without reaching the worker.
  if (engine.submit(make(100, {}, 4)).status != SubmitStatus::kInvalid ||
      engine.submit(make(101, prompts[0], 0)).status != SubmitStatus::kInvalid) {
    std::fprintf(stderr, "FAIL a bad request was not refused as invalid\n");
    ok = false;
  }

  // Each thread submits its share of the prompts, a little apart, while the others decode.
  std::vector<std::thread> submitters;
  for (int t = 0; t < kThreads; ++t) {
    submitters.emplace_back([&, t] {
      for (int p = t; p < n; p += kThreads) {
        const llm::SubmitResult r = engine.submit(make(p, prompts[p], kSteps));
        if (!r.ok()) std::fprintf(stderr, "FAIL submit %d refused: %s\n", p, r.message.c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    });
  }
  for (std::thread& th : submitters) th.join();

  drain.until(n);
  for (int p = 0; p < n; ++p) {
    const std::string label = "threaded request " + std::to_string(p);
    ok &= llm::testing::check_int_equal(label.c_str(), drain.got[p], alone[p]);
    if (drain.finish[p] != FinishReason::kLength) {
      std::fprintf(stderr, "FAIL %s did not finish by length\n", label.c_str());
      ok = false;
    }
  }

  {
    const llm::SchedulerStats st = engine.stats();
    if (st.running != 0 || st.waiting != 0 || st.kv_pages_used != 0 || st.decode_tokens != n * (kSteps - 1)) {
      std::fprintf(stderr, "FAIL stats after drain: %d running, %d waiting, %d pages, %lld decoded\n",
                   st.running, st.waiting, st.kv_pages_used, static_cast<long long>(st.decode_tokens));
      ok = false;
    }
  }

  // A live id is refused; a cancel ends a request with one cancelled event, and frees its id.
  if (!engine.submit(make(200, prompts[0], kSteps)).ok() ||
      engine.submit(make(200, prompts[1], 1)).status != SubmitStatus::kInvalid) {
    std::fprintf(stderr, "FAIL a live id was refused the first time or accepted twice\n");
    ok = false;
  }
  engine.cancel(200);
  drain.until(n + 1);
  if (drain.finish[200] != FinishReason::kCancelled || drain.got[200].size() >= alone[0].size()) {
    std::fprintf(stderr, "FAIL cancelled request finished '%s' with %zu tokens\n",
                 llm::finish_reason_name(drain.finish[200]), drain.got[200].size());
    ok = false;
  }
  if (!engine.submit(make(200, prompts[1], 1)).ok()) {
    std::fprintf(stderr, "FAIL a finished id could not be reused\n");
    ok = false;
  }
  drain.until(n + 1);

  // A full waiting line refuses with kBusy; everything it accepted still finishes.
  {
    llm::EngineOptions small = options;
    small.max_seqs = 1;
    small.max_waiting = 2;
    llm::AsyncEngine busy_engine(small);
    Drain busy_drain{busy_engine};
    int accepted = 0;
    int busy = 0;
    for (int i = 0; i < 10; ++i) {
      const llm::SubmitResult r = busy_engine.submit(make(i, prompts[i % n], kSteps));
      accepted += r.ok();
      busy += r.status == SubmitStatus::kBusy;
    }
    busy_drain.until(accepted);
    // Exactly one running plus max_waiting queued, unless one finished meanwhile.
    if (busy == 0 || accepted < 1 + small.max_waiting || accepted > 1 + small.max_waiting + 1 ||
        static_cast<int>(busy_drain.finish.size()) != accepted) {
      std::fprintf(stderr, "FAIL queue cap: %d accepted, %d busy, %zu finished\n", accepted, busy,
                   busy_drain.finish.size());
      ok = false;
    }
  }

  // A worker failure ends every live request with an error and refuses new work.
  {
    const std::size_t before = drain.finish.size();
    ok &= engine.submit(make(300, prompts[0], kSteps)).ok();
    ok &= engine.submit(make(301, prompts[2], kSteps)).ok();
    engine.fail_for_test("injected failure");
    drain.until(before + 2);
    if (drain.finish[300] != FinishReason::kError || drain.finish[301] != FinishReason::kError) {
      std::fprintf(stderr, "FAIL after a failure, requests ended '%s' and '%s', expected 'error'\n",
                   llm::finish_reason_name(drain.finish[300]), llm::finish_reason_name(drain.finish[301]));
      ok = false;
    }
    if (engine.healthy() || engine.failure() != "injected failure" ||
        engine.submit(make(302, prompts[0], 4)).status != SubmitStatus::kUnavailable) {
      std::fprintf(stderr, "FAIL a failed engine still looks healthy or accepts work\n");
      ok = false;
    }
  }

  engine.stop();
  if (!engine.poll(10).empty() && !engine.poll(10).empty()) {
    std::fprintf(stderr, "FAIL events kept arriving after stop\n");
    ok = false;
  }

  if (!ok) return EXIT_FAILURE;
  std::printf("PASS: %d requests from %d threads match decoding alone; refusals, cancel, queue cap "
              "and failures come back cleanly\n",
              n, kThreads);
  return EXIT_SUCCESS;
}
