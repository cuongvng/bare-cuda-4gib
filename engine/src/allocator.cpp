#include "llm/allocator.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

namespace llm {
namespace {

[[noreturn]] void fail(const std::string& message) {
  std::fprintf(stderr, "[llm] allocator: %s\n", message.c_str());
  std::exit(EXIT_FAILURE);
}

}  // namespace

void PagePool::allocate(std::size_t page_bytes, int num_pages) {
  if (page_bytes == 0 || page_bytes % sizeof(float) != 0) {
    fail("page_bytes must be a positive multiple of sizeof(float)");
  }
  if (num_pages <= 0) fail("num_pages must be positive");

  pool_.allocate(page_bytes * static_cast<std::size_t>(num_pages));
  page_bytes_ = page_bytes;
  high_water_mark_ = 0;
  in_use_.assign(num_pages, 0);

  // Filled backwards so pages come out in order 0, 1, 2, ...
  free_list_.resize(num_pages);
  for (int i = 0; i < num_pages; ++i) free_list_[i] = num_pages - 1 - i;
}

int PagePool::acquire() {
  if (free_list_.empty()) return -1;
  const int page = free_list_.back();
  free_list_.pop_back();
  in_use_[page] = 1;
  high_water_mark_ = std::max(high_water_mark_, bytes_in_use());
  return page;
}

bool PagePool::release(int page) {
  if (!is_valid(page) || !in_use_[page]) return false;
  in_use_[page] = 0;
  free_list_.push_back(page);
  return true;
}

void PagePool::shuffle_free_list(std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::shuffle(free_list_.begin(), free_list_.end(), rng);
}

float* PagePool::page_data(int page) {
  return const_cast<float*>(static_cast<const PagePool*>(this)->page_data(page));
}

const float* PagePool::page_data(int page) const {
  if (!is_valid(page)) {
    fail("page " + std::to_string(page) + " out of range, pool holds " +
         std::to_string(num_pages()));
  }
  return pool_.as<float>() + static_cast<std::size_t>(page) * (page_bytes_ / sizeof(float));
}

}  // namespace llm
