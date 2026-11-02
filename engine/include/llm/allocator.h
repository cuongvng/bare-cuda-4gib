#pragma once

#include <cstddef>
#include <vector>

#include "llm/device_buffer.h"

namespace llm {

// Fixed-size pages carved out of one cudaMalloc, handed out from a free list.
class PagePool {
 public:
  void allocate(std::size_t page_bytes, int num_pages);

  // Returns a page id, or -1 when every page is taken.
  [[nodiscard]] int acquire();

  // Returns false for an invalid id or a page that is already free.
  [[nodiscard]] bool release(int page);

  float* page_data(int page);
  const float* page_data(int page) const;

  int num_pages() const { return static_cast<int>(in_use_.size()); }
  int free_pages() const { return static_cast<int>(free_list_.size()); }
  int pages_in_use() const { return num_pages() - free_pages(); }

  std::size_t page_bytes() const { return page_bytes_; }
  std::size_t total_bytes() const { return pool_.bytes(); }
  std::size_t bytes_in_use() const { return page_bytes_ * pages_in_use(); }
  std::size_t high_water_mark() const { return high_water_mark_; }

 private:
  bool is_valid(int page) const { return page >= 0 && page < num_pages(); }

  DeviceBuffer pool_;
  std::size_t page_bytes_ = 0;
  std::size_t high_water_mark_ = 0;
  std::vector<int> free_list_;
  std::vector<char> in_use_;
};

}  // namespace llm
