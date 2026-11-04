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
};

// Runs the scheduler on a worker thread that owns the model; any thread may submit, cancel, poll.
class AsyncEngine {
 public:
  // Loads the model on the worker thread and returns once it is ready.
  explicit AsyncEngine(const EngineOptions& options);
  ~AsyncEngine();

  AsyncEngine(const AsyncEngine&) = delete;
  AsyncEngine& operator=(const AsyncEngine&) = delete;

  // Queues a request. Empty if accepted, else why not; ids must be unique among live requests.
  std::string submit(Request request);

  // Ends a request early; it then gets one last event with FinishReason::kCancelled.
  void cancel(std::int64_t id);

  // Waits up to timeout_ms for events; empty on timeout or once stopped.
  std::vector<TokenEvent> poll(int timeout_ms);

  // Stops the worker; waiting and running requests are dropped.
  void stop();

  const ModelConfig& config() const { return config_; }
  const EngineLimits& limits() const { return limits_; }

 private:
  void run(const EngineOptions& options);
  void publish(std::vector<TokenEvent> events);

  ModelConfig config_;
  EngineLimits limits_;

  // Inbox: requests and cancels for the worker, and the ids not yet finished.
  std::mutex in_mu_;
  std::condition_variable in_cv_;
  std::vector<Request> inbox_;
  std::vector<std::int64_t> cancels_;
  std::set<std::int64_t> live_;
  bool ready_ = false;
  bool stopping_ = false;

  // Outbox: events for whoever polls.
  std::mutex out_mu_;
  std::condition_variable out_cv_;
  std::deque<TokenEvent> outbox_;
  bool stopped_ = false;

  std::thread worker_;
};

}  // namespace llm
