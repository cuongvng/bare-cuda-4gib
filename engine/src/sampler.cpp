#include "llm/sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace llm {
namespace {

// Tokens sorted on the first top-p pass; a flat distribution that needs more falls back to all.
constexpr int kTopPFirstPass = 1024;

std::int32_t argmax(const float* logits, int vocab) {
  return static_cast<std::int32_t>(std::max_element(logits, logits + vocab) - logits);
}

}  // namespace

std::string check_sampling(const SamplingParams& p) {
  if (!(p.temperature >= 0.0f) || !std::isfinite(p.temperature)) {
    return "temperature must be a finite number >= 0";
  }
  if (!(p.top_p > 0.0f && p.top_p <= 1.0f)) return "top_p must be in (0, 1]";
  if (p.top_k < 0) return "top_k must be >= 0";
  return "";
}

std::uint64_t seed_for(const SamplingParams& p) {
  if (p.seed >= 0) return static_cast<std::uint64_t>(p.seed);
  std::random_device rd;
  return (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
}

std::int32_t sample_token(const float* logits, int vocab, const SamplingParams& p,
                          std::mt19937_64& rng) {
  if (p.greedy()) return argmax(logits, vocab);

  // Softmax weights at this temperature, unnormalized; subtracting the max keeps exp in range.
  const float max_logit = logits[argmax(logits, vocab)];
  const float inv_t = 1.0f / p.temperature;
  std::vector<float> w(vocab);
  double total = 0.0;
  for (int i = 0; i < vocab; ++i) {
    w[i] = std::exp((logits[i] - max_logit) * inv_t);
    total += w[i];
  }

  const int k = p.top_k > 0 ? std::min(p.top_k, vocab) : vocab;
  const bool top_p = p.top_p < 1.0f;
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  // No filtering: draw straight from all weights.
  if (k == vocab && !top_p) {
    double target = uniform(rng) * total;
    for (int i = 0; i < vocab; ++i) {
      target -= w[i];
      if (target < 0.0) return i;
    }
    return argmax(logits, vocab);
  }

  // Likeliest first, lower id on ties; only as many as the filters can keep get sorted.
  std::vector<std::int32_t> idx(vocab);
  std::iota(idx.begin(), idx.end(), 0);
  auto likelier = [&](std::int32_t a, std::int32_t b) { return w[a] > w[b] || (w[a] == w[b] && a < b); };

  int sorted = (top_p && k == vocab) ? std::min(kTopPFirstPass, vocab) : k;
  int keep = k;
  while (true) {
    std::partial_sort(idx.begin(), idx.begin() + sorted, idx.end(), likelier);
    if (!top_p) break;
    // top-p is measured on the top-k set, renormalized.
    double kept_total = total;
    if (k < vocab) {
      kept_total = 0.0;
      for (int i = 0; i < k; ++i) kept_total += w[idx[i]];
    }
    double cum = 0.0;
    keep = -1;
    for (int i = 0; i < sorted; ++i) {
      cum += w[idx[i]];
      if (cum >= p.top_p * kept_total) {
        keep = i + 1;
        break;
      }
    }
    if (keep > 0) break;
    if (sorted == k) {
      keep = k;  // rounding left the sum a hair short
      break;
    }
    sorted = k;
  }

  double kept = 0.0;
  for (int i = 0; i < keep; ++i) kept += w[idx[i]];
  double target = uniform(rng) * kept;
  for (int i = 0; i < keep; ++i) {
    target -= w[idx[i]];
    if (target < 0.0) return idx[i];
  }
  return idx[keep - 1];
}

}  // namespace llm
