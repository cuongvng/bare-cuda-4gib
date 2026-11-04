#include "llm/async_engine.h"

#include <chrono>
#include <exception>
#include <utility>

#include "llm/error.h"
#include "llm/model.h"

namespace llm {

const char* submit_status_name(SubmitStatus status) {
  switch (status) {
    case SubmitStatus::kOk: return "ok";
    case SubmitStatus::kInvalid: return "invalid";
    case SubmitStatus::kBusy: return "busy";
    case SubmitStatus::kUnavailable: return "unavailable";
  }
  return "";
}

AsyncEngine::AsyncEngine(const EngineOptions& options) : max_waiting_(options.max_waiting) {
  worker_ = std::thread([this, options] { run(options); });
  std::string load_error;
  {
    std::unique_lock<std::mutex> lock(in_mu_);
    in_cv_.wait(lock, [this] { return ready_; });
    load_error = failure_;
  }
  if (!load_error.empty()) {
    worker_.join();
    throw Error(load_error);
  }
}

AsyncEngine::~AsyncEngine() { stop(); }

SubmitResult AsyncEngine::submit(Request request) {
  const std::string error = Scheduler::check(limits_, request);
  if (!error.empty()) return {SubmitStatus::kInvalid, error};
  {
    std::lock_guard<std::mutex> lock(in_mu_);
    if (!failure_.empty()) return {SubmitStatus::kUnavailable, "engine failed: " + failure_};
    if (stopping_) return {SubmitStatus::kUnavailable, "engine is stopped"};
    // Live counts running, waiting and not yet seen by the worker.
    if (max_waiting_ > 0 && static_cast<int>(live_.size()) >= limits_.max_seqs + max_waiting_) {
      return {SubmitStatus::kBusy, std::to_string(live_.size()) + " requests already in flight"};
    }
    if (!live_.insert(request.id).second) {
      return {SubmitStatus::kInvalid, "request id " + std::to_string(request.id) + " is already in use"};
    }
    inbox_.push_back(std::move(request));
  }
  in_cv_.notify_one();
  return {};
}

void AsyncEngine::cancel(std::int64_t id) {
  {
    std::lock_guard<std::mutex> lock(in_mu_);
    cancels_.push_back(id);
  }
  in_cv_.notify_one();
}

std::vector<TokenEvent> AsyncEngine::poll(int timeout_ms) {
  std::unique_lock<std::mutex> lock(out_mu_);
  out_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                   [this] { return !outbox_.empty() || stopped_; });
  std::vector<TokenEvent> events(outbox_.begin(), outbox_.end());
  outbox_.clear();
  return events;
}

void AsyncEngine::stop() {
  {
    std::lock_guard<std::mutex> lock(in_mu_);
    stopping_ = true;
  }
  in_cv_.notify_one();
  if (worker_.joinable()) worker_.join();
  {
    std::lock_guard<std::mutex> lock(out_mu_);
    stopped_ = true;
  }
  out_cv_.notify_all();
}

SchedulerStats AsyncEngine::stats() const {
  std::lock_guard<std::mutex> lock(in_mu_);
  SchedulerStats s = stats_;
  s.waiting += static_cast<int>(inbox_.size());
  return s;
}

bool AsyncEngine::healthy() const {
  std::lock_guard<std::mutex> lock(in_mu_);
  return failure_.empty();
}

std::string AsyncEngine::failure() const {
  std::lock_guard<std::mutex> lock(in_mu_);
  return failure_;
}

void AsyncEngine::fail_for_test(const std::string& message) {
  {
    std::lock_guard<std::mutex> lock(in_mu_);
    injected_failure_ = message;
  }
  in_cv_.notify_one();
}

void AsyncEngine::publish(std::vector<TokenEvent> events) {
  if (events.empty()) return;
  {
    std::lock_guard<std::mutex> lock(in_mu_);
    for (const TokenEvent& e : events) {
      if (e.finish != FinishReason::kNone) live_.erase(e.id);
    }
  }
  {
    std::lock_guard<std::mutex> lock(out_mu_);
    outbox_.insert(outbox_.end(), events.begin(), events.end());
  }
  out_cv_.notify_all();
}

// Ends every live request with an error event; new submits are refused from here on.
void AsyncEngine::fail_all(const std::string& message) {
  std::vector<TokenEvent> events;
  {
    std::lock_guard<std::mutex> lock(in_mu_);
    failure_ = message;
    for (std::int64_t id : live_) events.push_back({id, -1, -1, FinishReason::kError});
    inbox_.clear();
    cancels_.clear();
  }
  publish(std::move(events));
}

void AsyncEngine::run(const EngineOptions& options) {
  try {
    // CUDA state is created and used on this thread only.
    Model model = Model::load(options.model_path, options.max_tokens, options.max_seqs,
                              options.kv_pages);
    Scheduler scheduler(model);
    {
      std::lock_guard<std::mutex> lock(in_mu_);
      config_ = model.config();
      limits_ = scheduler.limits();
      stats_ = scheduler.stats();
      ready_ = true;
    }
    in_cv_.notify_all();
    loop(scheduler);
  } catch (const std::exception& e) {
    // failure_ is set before ready_, so a constructor still waiting sees the load error.
    fail_all(e.what());
    {
      std::lock_guard<std::mutex> lock(in_mu_);
      ready_ = true;
    }
    in_cv_.notify_all();
  }
}

void AsyncEngine::loop(Scheduler& scheduler) {
  while (true) {
    std::vector<Request> requests;
    std::vector<std::int64_t> cancels;
    {
      std::unique_lock<std::mutex> lock(in_mu_);
      // Sleep only when there is nothing to decode and nothing new.
      in_cv_.wait(lock, [&] {
        return stopping_ || !inbox_.empty() || !cancels_.empty() || !scheduler.idle() ||
               !injected_failure_.empty();
      });
      if (stopping_) return;
      if (!injected_failure_.empty()) throw Error(injected_failure_);
      requests.swap(inbox_);
      cancels.swap(cancels_);
    }

    // Submits before cancels, so a cancel right after its submit always finds the request.
    for (Request& r : requests) scheduler.submit(std::move(r));
    std::vector<TokenEvent> events;
    for (std::int64_t id : cancels) {
      if (scheduler.cancel(id)) events.push_back({id, -1, -1, FinishReason::kCancelled});
    }
    publish(std::move(events));
    if (!scheduler.idle()) publish(scheduler.step());
    const SchedulerStats snapshot = scheduler.stats();
    std::lock_guard<std::mutex> lock(in_mu_);
    stats_ = snapshot;
  }
}

}  // namespace llm
