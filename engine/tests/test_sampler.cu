// Host-only: the sampler's draws must follow softmax(logits / T) restricted by top-k and top-p.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "llm/sampler.h"

namespace {

constexpr int kDraws = 40000;
constexpr double kTol = 0.015;  // about 4 standard errors at kDraws

llm::SamplingParams params(float t, float top_p = 1.0f, int top_k = 0) {
  llm::SamplingParams p;
  p.temperature = t;
  p.top_p = top_p;
  p.top_k = top_k;
  return p;
}

// Frequencies of the first expected.size() ids over kDraws draws; other ids must never appear.
bool check_freq(const char* label, const std::vector<float>& logits, const llm::SamplingParams& p,
                const std::vector<double>& expected) {
  std::mt19937_64 rng(1);
  std::vector<int> count(logits.size(), 0);
  for (int i = 0; i < kDraws; ++i) ++count[llm::sample_token(logits.data(), static_cast<int>(logits.size()), p, rng)];
  bool ok = true;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const double want = i < expected.size() ? expected[i] : 0.0;
    const double got = static_cast<double>(count[i]) / kDraws;
    if (std::fabs(got - want) > kTol || (want == 0.0 && count[i] > 0)) {
      std::fprintf(stderr, "FAIL %s: id %zu drawn %.4f of the time, expected %.4f\n", label, i, got, want);
      ok = false;
    }
  }
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  // Probabilities 0.5, 0.3, 0.2 at T = 1, plus one id that is effectively impossible.
  const std::vector<float> logits = {std::log(0.5f), std::log(0.3f), std::log(0.2f), -1e30f};

  ok &= check_freq("T=1", logits, params(1.0f), {0.5, 0.3, 0.2});
  // T = 0.5 squares the probabilities: 0.25, 0.09, 0.04 renormalized.
  ok &= check_freq("T=0.5", logits, params(0.5f), {0.25 / 0.38, 0.09 / 0.38, 0.04 / 0.38});
  // top-k 2 and top-p 0.7 both keep {0, 1}: 0.5 alone is short of 0.7, 0.5 + 0.3 reaches it.
  ok &= check_freq("top_k=2", logits, params(1.0f, 1.0f, 2), {0.625, 0.375});
  ok &= check_freq("top_p=0.7", logits, params(1.0f, 0.7f), {0.625, 0.375});
  // top-p on the top-k set: top-k 2 leaves 0.625 / 0.375, and top-p 0.6 then keeps id 0 only.
  ok &= check_freq("top_k=2 top_p=0.6", logits, params(1.0f, 0.6f, 2), {1.0});
  // Greedy settings always pick the max.
  ok &= check_freq("T=0", logits, params(0.0f), {1.0});
  ok &= check_freq("top_k=1", logits, params(2.0f, 1.0f, 1), {1.0});

  // A full-size vocab where the nucleus is far larger than the first sorting pass.
  {
    const int vocab = 32000;
    std::mt19937_64 gen(3);
    std::normal_distribution<float> normal(0.0f, 0.3f);
    std::vector<float> big(vocab);
    for (float& x : big) x = normal(gen);
    const llm::SamplingParams p = params(1.0f, 0.9f);

    // Brute-force nucleus: sort all weights, keep the fewest reaching 90%.
    std::vector<int> idx(vocab);
    for (int i = 0; i < vocab; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return big[a] > big[b] || (big[a] == big[b] && a < b); });
    double total = 0.0;
    for (float x : big) total += std::exp(x);
    std::size_t size = 0;
    double cum = 0.0;
    while (cum < 0.9 * total) cum += std::exp(big[idx[size++]]);
    // A few extra ids of margin, as float and double sums may disagree on the edge.
    const std::set<int> nucleus(idx.begin(), idx.begin() + std::min<std::size_t>(size + 3, vocab));
    std::mt19937_64 rng(5);
    for (int i = 0; i < 200; ++i) {
      const int t = llm::sample_token(big.data(), vocab, p, rng);
      if (!nucleus.count(t)) {
        std::fprintf(stderr, "FAIL big vocab: drew id %d outside the %zu-token nucleus\n", t, nucleus.size());
        ok = false;
        break;
      }
    }
    if (size <= 1024 || size + 3 >= static_cast<std::size_t>(vocab)) {
      std::fprintf(stderr, "FAIL big vocab: nucleus of %zu does not exercise the fallback\n", nucleus.size());
      ok = false;
    }
  }

  // The same seed gives the same draws.
  {
    std::mt19937_64 a(42), b(42);
    for (int i = 0; i < 100; ++i) {
      if (llm::sample_token(logits.data(), 4, params(1.0f), a) != llm::sample_token(logits.data(), 4, params(1.0f), b)) {
        std::fprintf(stderr, "FAIL same seed drew differently\n");
        ok = false;
        break;
      }
    }
  }

  if (llm::check_sampling(params(-1.0f)).empty() || llm::check_sampling(params(1.0f, 0.0f)).empty() ||
      llm::check_sampling(params(1.0f, 1.5f)).empty() || llm::check_sampling(params(1.0f, 1.0f, -1)).empty() ||
      llm::check_sampling(params(NAN)).empty() || !llm::check_sampling(params(0.7f, 0.9f, 40)).empty()) {
    std::fprintf(stderr, "FAIL check_sampling accepted bad params or refused good ones\n");
    ok = false;
  }

  if (!ok) return EXIT_FAILURE;
  std::printf("PASS: sampling follows temperature, top-k and top-p; seeds repeat\n");
  return EXIT_SUCCESS;
}
