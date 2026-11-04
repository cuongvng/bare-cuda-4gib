#pragma once

#include <cstdint>
#include <random>
#include <string>

namespace llm {

struct SamplingParams {
  float temperature = 0.0f;  // 0 is greedy
  float top_p = 1.0f;        // keep the fewest likeliest tokens whose probability reaches top_p
  int top_k = 0;             // keep the k likeliest tokens; 0 keeps all
  std::int64_t seed = -1;    // -1 draws a random seed

  bool greedy() const { return temperature <= 0.0f || top_k == 1; }
};

// Empty if the params are valid, else why not.
std::string check_sampling(const SamplingParams& params);

std::uint64_t seed_for(const SamplingParams& params);

// Draws one token id from a row of logits under temperature, top-k and top-p.
std::int32_t sample_token(const float* logits, int vocab, const SamplingParams& params,
                          std::mt19937_64& rng);

}  // namespace llm
