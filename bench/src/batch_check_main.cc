// `llm-batch-check`: batched decode against one-at-a-time decode on a real model.
//
// Prompts are slices of an offline-tokenized text, of mixed lengths and more of
// them than there are slots, so queueing and retiring run too. Each prompt is
// greedy-decoded alone with generate(), then all together with generate_batch().
//
// Batched decode multiplies with GEMM where alone decode uses GEMV, and the two
// sum in different orders, so a near-tie between the top two logits may break
// the other way. A divergence is reported with the gap between the two
// candidates' logits; only a gap above --tol counts as a failure.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "llm/model.h"

namespace {

constexpr std::int32_t kBosId = 1;
constexpr int kPromptLengths[] = {5, 17, 64, 120, 300, 8, 200, 37};  // text ids, BOS added
constexpr std::size_t kPromptSpacing = 1000;  // ids between the starts of two prompts' slices

void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s --model <file.llmbin> --ids <file.ids> [--tokens N] [--batch N] "
               "[--tol X]\n"
               "\n"
               "  --model   path to a converted .llmbin model\n"
               "  --ids     flat little-endian int32 token ids (tools/convert/tokenize_text.py)\n"
               "  --tokens  tokens to generate per prompt (default 64)\n"
               "  --batch   sequences decoded at once (default 4)\n"
               "  --tol     largest logit gap a divergence may have and still pass (default 0.01)\n",
               argv0);
}

std::vector<std::int32_t> read_ids(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    std::fprintf(stderr, "[llm] cannot open %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  const std::streamsize bytes = in.tellg();
  std::vector<std::int32_t> ids(static_cast<std::size_t>(bytes) / sizeof(std::int32_t));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(ids.data()), static_cast<std::streamsize>(ids.size() * sizeof(std::int32_t)));
  return ids;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string ids_path;
  int tokens = 64;
  int batch = 4;
  float tol = 0.01f;

  for (int i = 1; i < argc; ++i) {
    const bool has_value = i + 1 < argc;
    if (std::strcmp(argv[i], "--model") == 0 && has_value) {
      model_path = argv[++i];
    } else if (std::strcmp(argv[i], "--ids") == 0 && has_value) {
      ids_path = argv[++i];
    } else if (std::strcmp(argv[i], "--tokens") == 0 && has_value) {
      tokens = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--batch") == 0 && has_value) {
      batch = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--tol") == 0 && has_value) {
      tol = static_cast<float>(std::atof(argv[++i]));
    } else {
      usage(argv[0]);
      return EXIT_FAILURE;
    }
  }
  if (model_path.empty() || ids_path.empty() || tokens <= 0 || batch <= 0) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  const std::vector<std::int32_t> text = read_ids(ids_path);
  std::vector<std::vector<std::int32_t>> prompts;
  int longest = 0;
  for (std::size_t p = 0; p < std::size(kPromptLengths); ++p) {
    const std::size_t start = p * kPromptSpacing;
    const std::size_t len = static_cast<std::size_t>(kPromptLengths[p]);
    if (start + len > text.size()) {
      std::fprintf(stderr, "[llm] %s has %zu ids, too few for prompt %zu\n", ids_path.c_str(),
                   text.size(), p);
      return EXIT_FAILURE;
    }
    std::vector<std::int32_t> prompt = {kBosId};
    prompt.insert(prompt.end(), text.begin() + start, text.begin() + start + len);
    longest = std::max(longest, static_cast<int>(prompt.size()));
    prompts.push_back(std::move(prompt));
  }

  llm::Model model = llm::Model::load(model_path, longest + tokens, batch);
  std::printf("model      %s\n", model_path.c_str());
  std::printf("check      %zu prompts of %d-%d tokens, %d new tokens each, %d decoding at once\n",
              prompts.size(), static_cast<int>(prompts.front().size()), longest, tokens, batch);
  std::printf("kv pool    %d pages | max batch %d at %d tokens per sequence\n\n", model.kv_pages(),
              model.max_batch_size(), model.max_tokens());

  std::vector<std::vector<std::int32_t>> alone;
  for (const auto& prompt : prompts) alone.push_back(model.generate(prompt, tokens));
  const std::vector<std::vector<std::int32_t>> together = model.generate_batch(prompts, tokens);

  const std::size_t vocab = model.config().vocab_size;
  int exact = 0, near_ties = 0, failures = 0;
  for (std::size_t p = 0; p < prompts.size(); ++p) {
    const auto split = std::mismatch(alone[p].begin(), alone[p].end(), together[p].begin());
    if (split.first == alone[p].end()) {
      ++exact;
      std::printf("prompt %zu  %3zu tokens  match, %zu ids identical\n", p, prompts[p].size(),
                  alone[p].size());
      continue;
    }

    // Score both candidates from one fresh prefill of the shared prefix, a third, neutral path.
    const std::size_t step = static_cast<std::size_t>(split.first - alone[p].begin());
    std::vector<std::int32_t> prefix = prompts[p];
    prefix.insert(prefix.end(), alone[p].begin(), split.first);
    model.reset_cache();
    model.forward(prefix.data(), static_cast<int>(prefix.size()));
    const std::vector<float> logits = model.logits_host(static_cast<int>(prefix.size()));
    const float* row = logits.data() + (prefix.size() - 1) * vocab;
    const float gap = std::fabs(row[*split.first] - row[*split.second]);

    const bool tie = gap <= tol;
    tie ? ++near_ties : ++failures;
    std::printf("prompt %zu  %3zu tokens  %s at step %zu: alone %d, batched %d, logit gap %.5f\n",
                p, prompts[p].size(), tie ? "near-tie" : "DIVERGES", step, *split.first,
                *split.second, gap);
  }

  std::printf("\n%d match, %d split on a near-tie (gap <= %.3g), %d diverge\n", exact, near_ties,
              tol, failures);
  if (failures > 0) {
    std::printf("FAIL: batched decode disagrees with decoding alone beyond rounding\n");
    return EXIT_FAILURE;
  }
  std::printf("PASS: batched decode agrees with decoding alone\n");
  return EXIT_SUCCESS;
}
