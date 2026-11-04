#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "llm/model.h"

namespace llm {

enum class FinishReason { kNone, kStop, kLength, kCancelled };

const char* finish_reason_name(FinishReason reason);

struct Request {
  std::int64_t id = 0;
  std::vector<std::int32_t> prompt;
  int max_new_tokens = 16;
  std::vector<std::int32_t> stop_ids;  // generating any of these ends the request, e.g. EOS
};

// One generated token, or a request ending without one (token -1, e.g. cancelled).
struct TokenEvent {
  std::int64_t id = 0;
  std::int32_t token = -1;
  int index = -1;  // position among this request's generated tokens
  FinishReason finish = FinishReason::kNone;
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

  const EngineLimits& limits() const { return limits_; }

  // Queues a request; fails on one that check() refuses.
  void submit(Request request);

  // Drops a waiting or running request and frees its pages. False if the id is unknown.
  bool cancel(std::int64_t id);

  // Admits waiting requests in order while slots and pages allow (prefilling each),
  // then runs one decode pass over every running request.
  std::vector<TokenEvent> step();

  bool idle() const { return waiting_.empty() && running_.empty(); }
  int num_waiting() const { return static_cast<int>(waiting_.size()); }
  int num_running() const { return static_cast<int>(running_.size()); }

 private:
  struct Running {
    std::int64_t id;
    int slot;
    int pages;   // committed for the whole run
    int target;  // tokens to generate unless a stop id comes first
    int generated;
    std::int32_t last;
    std::vector<std::int32_t> stop_ids;
  };

  static int target_tokens(const EngineLimits& limits, const Request& r);
  static int pages_needed(const EngineLimits& limits, const Request& r);

  // Records the token; true if that finished the request, which is then retired.
  bool emit(Running& r, std::int32_t token, std::vector<TokenEvent>& events);
  void retire(const Running& r);

  Model& model_;
  EngineLimits limits_;
  std::deque<Request> waiting_;
  std::vector<Running> running_;
  std::vector<int> free_slots_;
  int committed_pages_ = 0;  // promised to running requests, held or still to be taken
};

}  // namespace llm
