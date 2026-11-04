#include "llm/scheduler.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "llm/error.h"

namespace llm {
namespace {

[[noreturn]] void fail(const std::string& message) { raise_error("[llm] scheduler: " + message); }

}  // namespace

const char* finish_reason_name(FinishReason reason) {
  switch (reason) {
    case FinishReason::kNone: return "";
    case FinishReason::kStop: return "stop";
    case FinishReason::kLength: return "length";
    case FinishReason::kCancelled: return "cancelled";
    case FinishReason::kTimeout: return "timeout";
    case FinishReason::kError: return "error";
  }
  return "";
}

Scheduler::Scheduler(Model& model) : model_(model) {
  limits_.max_tokens = model.max_tokens();
  limits_.max_seqs = model.max_seqs();
  limits_.kv_pages = model.kv_pages();
  for (int s = limits_.max_seqs - 1; s >= 0; --s) free_slots_.push_back(s);
  model_.reset_cache();
}

// Like generate(), a sequence stops early at max_tokens; its last token is never cached.
int Scheduler::target_tokens(const EngineLimits& limits, const Request& r) {
  return std::min(r.max_new_tokens, limits.max_tokens - static_cast<int>(r.prompt.size()) + 1);
}

int Scheduler::pages_needed(const EngineLimits& limits, const Request& r) {
  return KVCache::pages_for(static_cast<int>(r.prompt.size()) + target_tokens(limits, r) - 1);
}

std::string Scheduler::check(const EngineLimits& limits, const std::vector<std::int32_t>& prompt,
                             int max_new_tokens) {
  const int len = static_cast<int>(prompt.size());
  if (len == 0) return "empty prompt";
  if (max_new_tokens <= 0) return "max_new_tokens must be positive";
  if (len > limits.max_tokens) {
    return "prompt has " + std::to_string(len) + " tokens, the engine holds " +
           std::to_string(limits.max_tokens) + " per sequence";
  }
  Request r;
  r.prompt = prompt;
  r.max_new_tokens = max_new_tokens;
  const int pages = pages_needed(limits, r);
  if (pages > limits.kv_pages) {
    return "request needs " + std::to_string(pages) + " KV pages, the pool holds " +
           std::to_string(limits.kv_pages);
  }
  return "";
}

std::string Scheduler::check(const EngineLimits& limits, const Request& request) {
  const std::string error = check(limits, request.prompt, request.max_new_tokens);
  return error.empty() ? check_sampling(request.sampling) : error;
}

void Scheduler::submit(Request request) {
  const std::string error = check(limits_, request);
  if (!error.empty()) fail("request " + std::to_string(request.id) + ": " + error);
  const Clock::time_point deadline = request.timeout_ms > 0
                                         ? Clock::now() + std::chrono::milliseconds(request.timeout_ms)
                                         : Clock::time_point::max();
  waiting_.push_back({std::move(request), deadline});
}

bool Scheduler::cancel(std::int64_t id) {
  for (auto it = waiting_.begin(); it != waiting_.end(); ++it) {
    if (it->request.id == id) {
      waiting_.erase(it);
      return true;
    }
  }
  for (auto it = running_.begin(); it != running_.end(); ++it) {
    if (it->id == id) {
      retire(*it);
      running_.erase(it);
      return true;
    }
  }
  return false;
}

std::int32_t Scheduler::choose(Running& r, std::int32_t argmax, int logits_row) {
  if (r.sampling.greedy()) return argmax;
  const std::vector<float> logits = model_.logits_row(logits_row);
  return sample_token(logits.data(), static_cast<int>(logits.size()), r.sampling, r.rng);
}

bool Scheduler::emit(Running& r, std::int32_t token, std::vector<TokenEvent>& events) {
  TokenEvent e;
  e.id = r.id;
  e.token = token;
  e.index = r.generated++;
  r.last = token;
  if (std::find(r.stop_ids.begin(), r.stop_ids.end(), token) != r.stop_ids.end()) {
    e.finish = FinishReason::kStop;
  } else if (r.generated == r.target) {
    e.finish = FinishReason::kLength;
  }
  events.push_back(e);
  if (e.finish == FinishReason::kNone) return false;
  retire(r);
  return true;
}

void Scheduler::retire(const Running& r) {
  model_.release_slot(r.slot);
  free_slots_.push_back(r.slot);
  committed_pages_ -= r.pages;
}

void Scheduler::expire(std::vector<TokenEvent>& events) {
  const Clock::time_point now = Clock::now();
  auto timed_out = [&](std::int64_t id) { events.push_back({id, -1, -1, FinishReason::kTimeout}); };
  for (auto it = waiting_.begin(); it != waiting_.end();) {
    if (it->deadline > now) {
      ++it;
      continue;
    }
    timed_out(it->request.id);
    it = waiting_.erase(it);
  }
  std::vector<Running> still_running;
  for (Running& r : running_) {
    if (r.deadline > now) {
      still_running.push_back(std::move(r));
      continue;
    }
    timed_out(r.id);
    retire(r);
  }
  running_ = std::move(still_running);
}

std::vector<TokenEvent> Scheduler::step() {
  std::vector<TokenEvent> events;
  expire(events);

  // Admit in order; a request that does not fit yet blocks the ones behind it.
  while (!waiting_.empty() && !free_slots_.empty() &&
         committed_pages_ + pages_needed(limits_, waiting_.front().request) <= limits_.kv_pages) {
    const Clock::time_point deadline = waiting_.front().deadline;
    Request req = std::move(waiting_.front().request);
    waiting_.pop_front();
    Running r{req.id, free_slots_.back(), pages_needed(limits_, req), target_tokens(limits_, req),
              0, 0, std::move(req.stop_ids), req.sampling, std::mt19937_64(seed_for(req.sampling)),
              deadline};
    free_slots_.pop_back();
    committed_pages_ += r.pages;

    const int len = static_cast<int>(req.prompt.size());
    model_.forward(req.prompt.data(), len, r.slot);
    const std::int32_t token = choose(r, model_.argmax_last(len), len - 1);
    if (!emit(r, token, events)) running_.push_back(std::move(r));
  }
  if (running_.empty()) return events;

  std::vector<std::int32_t> ids;
  std::vector<int> slots;
  for (const Running& r : running_) {
    ids.push_back(r.last);
    slots.push_back(r.slot);
  }
  const int n = static_cast<int>(running_.size());
  model_.forward_decode(ids.data(), slots.data(), n);
  const std::vector<std::int32_t> next = model_.argmax_rows(n);

  std::vector<Running> still_running;
  for (int b = 0; b < n; ++b) {
    const std::int32_t token = choose(running_[b], next[b], b);
    if (!emit(running_[b], token, events)) still_running.push_back(std::move(running_[b]));
  }
  running_ = std::move(still_running);
  return events;
}

}  // namespace llm
