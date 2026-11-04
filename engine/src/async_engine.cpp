#include "llm/async_engine.h"

#include <chrono>
#include <utility>

#include "llm/model.h"

namespace llm {

AsyncEngine::AsyncEngine(const EngineOptions& options) {
  worker_ = std::thread([this, options] { run(options); });
  std::unique_lock<std::mutex> lock(in_mu_);
  in_cv_.wait(lock, [this] { return ready_; });
}

AsyncEngine::~AsyncEngine() { stop(); }

std::string AsyncEngine::submit(Request request) {
  const std::string error = Scheduler::check(limits_, request);
  if (!error.empty()) return error;
  {
    std::lock_guard<std::mutex> lock(in_mu_);
    if (stopping_) return "engine is stopped";
    if (!live_.insert(request.id).second) {
      return "request id " + std::to_string(request.id) + " is already in use";
    }
    inbox_.push_back(std::move(request));
  }
  in_cv_.notify_one();
  return "";
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

void AsyncEngine::run(const EngineOptions& options) {
  // CUDA state is created and used on this thread only.
  Model model = Model::load(options.model_path, options.max_tokens, options.max_seqs,
                            options.kv_pages);
  Scheduler scheduler(model);
  {
    std::lock_guard<std::mutex> lock(in_mu_);
    config_ = model.config();
    limits_ = scheduler.limits();
    ready_ = true;
  }
  in_cv_.notify_all();

  while (true) {
    std::vector<Request> requests;
    std::vector<std::int64_t> cancels;
    {
      std::unique_lock<std::mutex> lock(in_mu_);
      // Sleep only when there is nothing to decode and nothing new.
      in_cv_.wait(lock, [&] {
        return stopping_ || !inbox_.empty() || !cancels_.empty() || !scheduler.idle();
      });
      if (stopping_) return;
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
  }
}

}  // namespace llm
