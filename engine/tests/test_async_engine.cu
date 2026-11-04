// Requests submitted from several threads at once, while the worker decodes, must each get
// the ids they get decoded alone; refusals and cancels must come back cleanly.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "llm/async_engine.h"
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

}  // namespace

int main(int argc, char** argv) {
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

  llm::EngineOptions options;
  options.model_path = argv[1];
  options.max_tokens = max_tokens;
  options.max_seqs = kMaxSeqs;
  llm::AsyncEngine engine(options);
  bool ok = true;

  // Refusals come back at once, without reaching the worker.
  if (engine.submit(make(100, {}, 4)).empty() || engine.submit(make(101, prompts[0], 0)).empty()) {
    std::fprintf(stderr, "FAIL a bad request was accepted\n");
    ok = false;
  }

  // Each thread submits its share of the prompts, a little apart, while the others decode.
  std::vector<std::thread> submitters;
  for (int t = 0; t < kThreads; ++t) {
    submitters.emplace_back([&, t] {
      for (int p = t; p < n; p += kThreads) {
        const std::string error = engine.submit(make(p, prompts[p], kSteps));
        if (!error.empty()) std::fprintf(stderr, "FAIL submit %d refused: %s\n", p, error.c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    });
  }
  for (std::thread& th : submitters) th.join();

  std::map<std::int64_t, std::vector<std::int32_t>> got;
  std::map<std::int64_t, llm::FinishReason> finish;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  auto drain = [&](std::size_t done_count) {
    while (finish.size() < done_count && std::chrono::steady_clock::now() < deadline) {
      for (const llm::TokenEvent& e : engine.poll(100)) {
        if (e.token >= 0) got[e.id].push_back(e.token);
        if (e.finish != llm::FinishReason::kNone) finish[e.id] = e.finish;
      }
    }
  };
  drain(n);
  for (int p = 0; p < n; ++p) {
    const std::string label = "threaded request " + std::to_string(p);
    ok &= llm::testing::check_int_equal(label.c_str(), got[p], alone[p]);
    if (finish[p] != llm::FinishReason::kLength) {
      std::fprintf(stderr, "FAIL %s did not finish by length\n", label.c_str());
      ok = false;
    }
  }

  // A live id is refused; a cancel ends a request with one cancelled event, and frees its id.
  if (!engine.submit(make(200, prompts[0], kSteps)).empty() ||
      engine.submit(make(200, prompts[1], 1)).empty()) {
    std::fprintf(stderr, "FAIL a live id was refused the first time or accepted twice\n");
    ok = false;
  }
  engine.cancel(200);
  drain(n + 1);
  if (finish[200] != llm::FinishReason::kCancelled || got[200].size() >= alone[0].size()) {
    std::fprintf(stderr, "FAIL cancelled request finished '%s' with %zu tokens\n",
                 llm::finish_reason_name(finish[200]), got[200].size());
    ok = false;
  }
  if (!engine.submit(make(200, prompts[1], 1)).empty()) {
    std::fprintf(stderr, "FAIL a finished id could not be reused\n");
    ok = false;
  }

  engine.stop();
  if (!engine.poll(10).empty() && !engine.poll(10).empty()) {
    std::fprintf(stderr, "FAIL events kept arriving after stop\n");
    ok = false;
  }

  if (!ok) return EXIT_FAILURE;
  std::printf("PASS: %d requests from %d threads match decoding alone; refusals and cancel work\n",
              n, kThreads);
  return EXIT_SUCCESS;
}
