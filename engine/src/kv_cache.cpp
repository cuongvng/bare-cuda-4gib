#include "llm/kv_cache.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace llm {
namespace {

[[noreturn]] void fail(const std::string& message) {
  std::fprintf(stderr, "[llm] kv_cache: %s\n", message.c_str());
  std::exit(EXIT_FAILURE);
}

}  // namespace

std::size_t KVCache::layer_floats() const {
  return static_cast<std::size_t>(kKVTokensPerPage) * kv_dim_;
}

void KVCache::allocate(const ModelConfig& config, int max_tokens) {
  if (max_tokens <= 0) fail("max_tokens must be positive");

  max_tokens_ = max_tokens;
  num_layers_ = static_cast<int>(config.num_layers);
  kv_dim_ = static_cast<int>(config.num_kv_heads * config.head_dim);
  length_ = 0;

  const int num_pages = (max_tokens + kKVTokensPerPage - 1) / kKVTokensPerPage;
  const std::size_t page_floats = 2 * static_cast<std::size_t>(num_layers_) * layer_floats();
  pool_.allocate(page_floats * sizeof(float), num_pages);

  pages_.clear();
  page_table_.allocate(num_pages * sizeof(std::int32_t));
}

void KVCache::reserve(int n_new) {
  if (n_new < 0) fail("reserve called with a negative count");

  const int needed = (length_ + n_new + kKVTokensPerPage - 1) / kKVTokensPerPage;
  while (static_cast<int>(pages_.size()) < needed) {
    const std::int32_t page = pool_.acquire();
    if (page < 0) fail("out of pages, pool holds " + std::to_string(pool_.num_pages()));

    // Only the new entry goes to the GPU, so decode pays nothing on most steps.
    std::int32_t* entry = page_table_.as<std::int32_t>() + pages_.size();
    LLM_CUDA_CHECK(cudaMemcpy(entry, &page, sizeof(page), cudaMemcpyHostToDevice));
    pages_.push_back(page);
  }
}

void KVCache::write(int layer, const float* new_k, const float* new_v, int n_new) {
  const KVPages k = pages_at(layer);
  const std::size_t v_offset = static_cast<std::size_t>(num_layers_) * layer_floats();

  // One launch per page touched: the write kernel only knows a single slab.
  int done = 0;
  while (done < n_new) {
    const int token = length_ + done;
    const int row = token & (kKVTokensPerPage - 1);
    const int count = std::min(n_new - done, kKVTokensPerPage - row);

    float* page = pool_.page_data(pages_[token >> kKVTokenShift]);
    float* k_rows = page + k.layer_offset;
    const std::size_t new_offset = static_cast<std::size_t>(done) * kv_dim_;
    launch_kv_cache_write(new_k + new_offset, new_v + new_offset, k_rows, k_rows + v_offset, count,
                          row, kv_dim_);
    done += count;
  }
}

KVPages KVCache::pages_at(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    fail("layer " + std::to_string(layer) + " out of range, cache holds " +
         std::to_string(num_layers_));
  }
  KVPages pages;
  pages.pool = pool_.page_data(0);
  pages.page_table = page_table_.as<std::int32_t>();
  pages.page_stride = pool_.page_bytes() / sizeof(float);
  pages.layer_offset = static_cast<std::size_t>(layer) * layer_floats();
  pages.token_shift = kKVTokenShift;
  return pages;
}

KVPages KVCache::k_pages(int layer) const { return pages_at(layer); }

KVPages KVCache::v_pages(int layer) const {
  KVPages pages = pages_at(layer);
  pages.layer_offset += static_cast<std::size_t>(num_layers_) * layer_floats();
  return pages;
}

void KVCache::advance(int n_new) {
  if (n_new < 0) fail("advance called with a negative count");
  if (length_ + n_new > max_tokens_) {
    // The context limit, reported where it is actually hit.
    fail("sequence of " + std::to_string(length_ + n_new) + " exceeds cache capacity " +
         std::to_string(max_tokens_));
  }
  length_ += n_new;
}

void KVCache::reset() {
  for (const std::int32_t page : pages_) {
    if (!pool_.release(page)) fail("page " + std::to_string(page) + " was already free");
  }
  pages_.clear();
  length_ = 0;
}

}  // namespace llm
