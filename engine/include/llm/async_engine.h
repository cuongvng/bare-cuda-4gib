#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "llm/config.h"
#include "llm/scheduler.h"

namespace llm {

struct EngineOptions {
  std::string model_path;
  int max_tokens = 1024;  // per sequence, prompt plus generated
  int max_seqs = 8;
  int kv_pages = 0;       // 0 fits the pool to free VRAM
  int max_waiting = 64;   // beyond max_seqs running; more live requests get kBusy. 0 is unlimited
};

enum class SubmitStatus {
  kOk,
  kInvalid,      // the request itself is wrong, e.g. too long
  kBusy,         // the waiting line is full; try again later
  kUnavailable,  // the engine has stopped or failed
};

const char* submit_status_name(SubmitStatus status);

struct SubmitResult {
  SubmitStatus status = SubmitStatus::kOk;
  std::string message;

  bool ok() const { return status == SubmitStatus::kOk; }
};

// Runs the scheduler on a worker thread that owns the model; any thread may submit, cancel, poll.
// If the worker hits an error, every live request ends with FinishReason::kError and the
// engine refuses new work; healthy() then turns false.
class AsyncEngine {
 public:
  // Loads the model on the worker thread and returns once it is ready; throws Error if loading fails.
  explicit AsyncEngine(const EngineOptions& options);
  ~AsyncEngine();

  AsyncEngine(const AsyncEngine&) = delete;
  AsyncEngine& operator=(const AsyncEngine&) = delete;

  // Queues a request; ids must be unique among live requests.
  SubmitResult submit(Request request);

  // Ends a request early; it then gets one last event with FinishReason::kCancelled.
  void cancel(std::int64_t id);

  // Waits up to timeout_ms for events; empty on timeout or once stopped.
  std::vector<TokenEvent> poll(int timeout_ms);

  // Stops the worker; waiting and running requests are dropped.
  void stop();

  bool healthy() const;
  std::string failure() const;  // why the engine failed; empty while healthy

  // Makes the worker throw on its next loop, to exercise the failure path in tests.
  void fail_for_test(const std::string& message);

  const ModelConfig& config() const { return config_; }
  const EngineLimits& limits() const { return limits_; }

 private:
  void run(const EngineOptions& options);
  void loop(Scheduler& scheduler);
  void publish(std::vector<TokenEvent> events);
  void fail_all(const std::string& message);

  ModelConfig config_;
  EngineLimits limits_;
  int max_waiting_ = 0;

  // Inbox: requests and cancels for the worker, and the ids not yet finished.
  mutable std::mutex in_mu_;
  std::condition_variable in_cv_;
  std::vector<Request> inbox_;
  std::vector<std::int64_t> cancels_;
  std::set<std::int64_t> live_;
  bool ready_ = false;
  bool stopping_ = false;
  std::string failure_;
  std::string injected_failure_;

  // Outbox: events for whoever polls.
  std::mutex out_mu_;
  std::condition_variable out_cv_;
  std::deque<TokenEvent> outbox_;
  bool stopped_ = false;

  std::thread worker_;
};

}  // namespace llm
