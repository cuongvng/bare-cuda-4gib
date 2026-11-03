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

void KVCache::allocate(const ModelConfig& config, int max_tokens, int max_seqs) {
  if (max_tokens <= 0) fail("max_tokens must be positive");
  if (max_seqs <= 0) fail("max_seqs must be positive");

  max_tokens_ = max_tokens;
  max_pages_per_seq_ = (max_tokens + kKVTokensPerPage - 1) / kKVTokensPerPage;
  num_layers_ = static_cast<int>(config.num_layers);
  kv_dim_ = static_cast<int>(config.num_kv_heads * config.head_dim);

  // Enough for every slot to reach max_tokens at once.
  const int num_pages = max_seqs * max_pages_per_seq_;
  const std::size_t page_floats = 2 * static_cast<std::size_t>(num_layers_) * layer_floats();
  pool_.allocate(page_floats * sizeof(float), num_pages);

  slots_.assign(max_seqs, Slot{});
  page_table_.allocate(static_cast<std::size_t>(num_pages) * sizeof(std::int32_t));
}

const KVCache::Slot& KVCache::slot_at(int slot) const {
  if (slot < 0 || slot >= max_seqs()) {
    fail("slot " + std::to_string(slot) + " out of range, cache holds " +
         std::to_string(max_seqs()));
  }
  return slots_[slot];
}

KVCache::Slot& KVCache::slot_at(int slot) {
  return const_cast<Slot&>(static_cast<const KVCache&>(*this).slot_at(slot));
}

void KVCache::reserve(int slot, int n_new) {
  if (n_new < 0) fail("reserve called with a negative count");
  Slot& s = slot_at(slot);

  const int needed = (s.length + n_new + kKVTokensPerPage - 1) / kKVTokensPerPage;
  if (needed > max_pages_per_seq_) {
    fail("slot " + std::to_string(slot) + " needs " + std::to_string(needed) +
         " pages, its table row holds " + std::to_string(max_pages_per_seq_));
  }
  while (static_cast<int>(s.pages.size()) < needed) {
    const std::int32_t page = pool_.acquire();
    if (page < 0) fail("out of pages, pool holds " + std::to_string(pool_.num_pages()));

    // Only the new entry goes to the GPU, so decode pays nothing on most steps.
    std::int32_t* entry = page_table_.as<std::int32_t>() +
                          static_cast<std::size_t>(slot) * max_pages_per_seq_ + s.pages.size();
    LLM_CUDA_CHECK(cudaMemcpy(entry, &page, sizeof(page), cudaMemcpyHostToDevice));
    s.pages.push_back(page);
  }
}

void KVCache::write(int slot, int layer, const float* new_k, const float* new_v, int n_new) {
  const Slot& s = slot_at(slot);
  const KVPages k = pages_at(slot, layer);
  const std::size_t v_offset = static_cast<std::size_t>(num_layers_) * layer_floats();

  // One launch per page touched: the write kernel only knows a single slab.
  int done = 0;
  while (done < n_new) {
    const int token = s.length + done;
    const int row = token & (kKVTokensPerPage - 1);
    const int count = std::min(n_new - done, kKVTokensPerPage - row);

    float* page = pool_.page_data(s.pages[token >> kKVTokenShift]);
    float* k_rows = page + k.layer_offset;
    const std::size_t new_offset = static_cast<std::size_t>(done) * kv_dim_;
    launch_kv_cache_write(new_k + new_offset, new_v + new_offset, k_rows, k_rows + v_offset, count,
                          row, kv_dim_);
    done += count;
  }
}

KVPages KVCache::pages_at(int slot, int layer) const {
  slot_at(slot);
  if (layer < 0 || layer >= num_layers_) {
    fail("layer " + std::to_string(layer) + " out of range, cache holds " +
         std::to_string(num_layers_));
  }
  KVPages pages;
  pages.pool = pool_.page_data(0);
  pages.page_table = page_table() + static_cast<std::size_t>(slot) * max_pages_per_seq_;
  pages.page_stride = pool_.page_bytes() / sizeof(float);
  pages.layer_offset = static_cast<std::size_t>(layer) * layer_floats();
  pages.token_shift = kKVTokenShift;
  return pages;
}

KVPages KVCache::k_pages(int slot, int layer) const { return pages_at(slot, layer); }

KVPages KVCache::v_pages(int slot, int layer) const {
  KVPages pages = pages_at(slot, layer);
  pages.layer_offset += static_cast<std::size_t>(num_layers_) * layer_floats();
  return pages;
}

KVPages KVCache::k_pages_batch(int layer, const std::int32_t* row_slot) const {
  KVPages pages = pages_at(0, layer);
  pages.table_stride = max_pages_per_seq_;
  pages.row_slot = row_slot;
  return pages;
}

KVPages KVCache::v_pages_batch(int layer, const std::int32_t* row_slot) const {
  KVPages pages = v_pages(0, layer);
  pages.table_stride = max_pages_per_seq_;
  pages.row_slot = row_slot;
  return pages;
}

void KVCache::advance(int slot, int n_new) {
  if (n_new < 0) fail("advance called with a negative count");
  Slot& s = slot_at(slot);
  if (s.length + n_new > max_tokens_) {
    // The context limit, reported where it is actually hit.
    fail("sequence of " + std::to_string(s.length + n_new) + " exceeds cache capacity " +
         std::to_string(max_tokens_));
  }
  s.length += n_new;
}

void KVCache::reset(int slot) {
  Slot& s = slot_at(slot);
  for (const std::int32_t page : s.pages) {
    if (!pool_.release(page)) fail("page " + std::to_string(page) + " was already free");
  }
  s.pages.clear();
  s.length = 0;
}

void KVCache::reset_all() {
  for (int slot = 0; slot < max_seqs(); ++slot) reset(slot);
}

}  // namespace llm
