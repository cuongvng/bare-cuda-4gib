// `llm-infer`: load a .llmbin model, run a prompt of pre-tokenized ids through
// it, print the generated ids.
//
// Ids in, ids out: there is no tokenizer in the engine by design -- text goes
// through the offline Python step, and the id -> string table is a printing
// concern for later.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "llm/model.h"

namespace {

void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s --model <file.llmbin> --prompt <id,id,id> [--tokens N] [--max-seq N]\n"
               "\n"
               "  --model    path to a converted .llmbin model\n"
               "  --prompt   comma-separated int32 token ids (offline-tokenized)\n"
               "  --tokens   number of tokens to generate (default 16)\n"
               "  --max-seq  workspace size in tokens; must cover prompt + generated\n",
               argv0);
}

std::vector<std::int32_t> parse_ids(const std::string& text) {
  std::vector<std::int32_t> ids;
  std::size_t pos = 0;
  while (pos <= text.size()) {
    const std::size_t comma = text.find(',', pos);
    const std::string piece = text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    if (!piece.empty()) ids.push_back(static_cast<std::int32_t>(std::strtol(piece.c_str(), nullptr, 10)));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return ids;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string prompt_text;
  int new_tokens = 16;
  int max_seq = 0;

  for (int i = 1; i < argc; ++i) {
    const bool has_value = i + 1 < argc;
    if (std::strcmp(argv[i], "--model") == 0 && has_value) {
      model_path = argv[++i];
    } else if (std::strcmp(argv[i], "--prompt") == 0 && has_value) {
      prompt_text = argv[++i];
    } else if (std::strcmp(argv[i], "--tokens") == 0 && has_value) {
      new_tokens = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--max-seq") == 0 && has_value) {
      max_seq = std::atoi(argv[++i]);
    } else {
      usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  if (model_path.empty() || prompt_text.empty()) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  const std::vector<std::int32_t> prompt = parse_ids(prompt_text);
  if (prompt.empty()) {
    std::fprintf(stderr, "[llm] no token ids parsed from --prompt\n");
    return EXIT_FAILURE;
  }

  if (max_seq <= 0) max_seq = static_cast<int>(prompt.size()) + new_tokens;

  llm::Model model = llm::Model::load(model_path, max_seq);
  std::printf(
      "[llm] %u layers, hidden %u, vocab %u | weights %.1f MiB, workspace %.1f MiB, cache %.1f "
      "MiB (%d pages, max batch %d)\n",
      model.config().num_layers, model.config().hidden_size, model.config().vocab_size,
      model.weight_bytes() / (1024.0 * 1024.0), model.workspace_bytes() / (1024.0 * 1024.0),
      model.cache_bytes() / (1024.0 * 1024.0), model.kv_pages(), model.max_batch_size());

  const std::vector<std::int32_t> generated = model.generate(prompt, new_tokens);

  std::printf("prompt:    ");
  for (std::int32_t id : prompt) std::printf("%d ", id);
  std::printf("\ngenerated: ");
  for (std::int32_t id : generated) std::printf("%d ", id);
  std::printf("\n");
  return EXIT_SUCCESS;
}
