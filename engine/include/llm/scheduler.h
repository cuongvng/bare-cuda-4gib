#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <random>
#include <string>
#include <vector>

#include "llm/model.h"
#include "llm/sampler.h"

namespace llm {

enum class FinishReason { kNone, kStop, kLength, kCancelled, kTimeout, kError };

const char* finish_reason_name(FinishReason reason);

struct Request {
  std::int64_t id = 0;
  std::vector<std::int32_t> prompt;
  int max_new_tokens = 16;
  std::vector<std::int32_t> stop_ids;  // generating any of these ends the request, e.g. EOS
  SamplingParams sampling;             // greedy by default
  int timeout_ms = 0;                  // from submit, waiting plus running; 0 never times out
};

// One generated token, or a request ending without one (token -1, e.g. cancelled).
struct TokenEvent {
  std::int64_t id = 0;
  std::int32_t token = -1;
  int index = -1;  // position among this request's generated tokens
  FinishReason finish = FinishReason::kNone;
};

// Counters since the scheduler started, plus a snapshot of what it holds now.
struct SchedulerStats {
  int running = 0;
  int waiting = 0;
  int kv_pages_used = 0;       // held by running requests now
  int kv_pages_committed = 0;  // promised to running requests for their whole run
  int kv_pages_total = 0;
  std::int64_t steps = 0;           // decode passes
  std::int64_t prefill_tokens = 0;  // prompt tokens run through the model
  std::int64_t decode_tokens = 0;   // rows decoded, one per running request per pass
  double busy_seconds = 0.0;        // wall time spent inside step()
};

// What the engine can hold; fixed once the model is loaded.
struct EngineLimits {
  int max_tokens = 0;  // per sequence, prompt plus generated
  int max_seqs = 0;
  int kv_pages = 0;
};

// Continuous batching on one model: requests join while others decode and leave when done.
// Not thread-safe; one thread calls everything.
class Scheduler {
 public:
  explicit Scheduler(Model& model);

  // Empty if the request can ever run, else why not.
  static std::string check(const EngineLimits& limits, const std::vector<std::int32_t>& prompt,
                           int max_new_tokens);

  // Like above, and also checks the request's sampling params.
  static std::string check(const EngineLimits& limits, const Request& request);

  const EngineLimits& limits() const { return limits_; }

  // Queues a request; fails on one that check() refuses.
  void submit(Request request);

  // Drops a waiting or running request and frees its pages. False if the id is unknown.
  bool cancel(std::int64_t id);

  // Ends requests past their deadline, admits waiting ones in order while slots and pages allow
  // (prefilling each), then runs one decode pass over every running request.
  std::vector<TokenEvent> step();

  bool idle() const { return waiting_.empty() && running_.empty(); }
  int num_waiting() const { return static_cast<int>(waiting_.size()); }
  int num_running() const { return static_cast<int>(running_.size()); }
  SchedulerStats stats() const;

 private:
  using Clock = std::chrono::steady_clock;

  struct Waiting {
    Request request;
    Clock::time_point deadline;
  };

  struct Running {
    std::int64_t id;
    int slot;
    int pages;   // committed for the whole run
    int target;  // tokens to generate unless a stop id comes first
    int generated;
    std::int32_t last;
    std::vector<std::int32_t> stop_ids;
    SamplingParams sampling;
    std::mt19937_64 rng;  // per request, so a seed gives the same tokens alone or batched
    Clock::time_point deadline;
  };

  std::vector<TokenEvent> run_step();

  // Drops every request whose deadline has passed, with a timeout event each.
  void expire(std::vector<TokenEvent>& events);

  static int target_tokens(const EngineLimits& limits, const Request& r);
  static int pages_needed(const EngineLimits& limits, const Request& r);

  // The GPU argmax for greedy requests, else a draw from the given logits row.
  std::int32_t choose(Running& r, std::int32_t argmax, int logits_row);

  // Records the token; true if that finished the request, which is then retired.
  bool emit(Running& r, std::int32_t token, std::vector<TokenEvent>& events);
  void retire(const Running& r);

  Model& model_;
  EngineLimits limits_;
  std::deque<Waiting> waiting_;
  std::vector<Running> running_;
  std::vector<int> free_slots_;
  int committed_pages_ = 0;  // promised to running requests, held or still to be taken
  SchedulerStats totals_;     // only the counters are kept up to date here
};

}  // namespace llm
