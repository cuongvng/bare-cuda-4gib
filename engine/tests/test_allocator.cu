// Checks PagePool's bookkeeping, then that its pages really are separate GPU memory.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <vector>

#include "llm/allocator.h"

namespace {

constexpr int kPages = 32;
constexpr int kFloatsPerPage = 256;
constexpr std::size_t kPageBytes = kFloatsPerPage * sizeof(float);
constexpr int kStressSteps = 100000;

bool expect(bool condition, const char* what) {
  if (!condition) std::fprintf(stderr, "FAIL %s\n", what);
  return condition;
}

bool test_exhaust_and_refill() {
  llm::PagePool pool;
  pool.allocate(kPageBytes, kPages);
  bool ok = expect(pool.total_bytes() == kPageBytes * kPages, "pool size");

  std::set<int> seen;
  for (int i = 0; i < kPages; ++i) seen.insert(pool.acquire());
  ok &= expect(seen.size() == kPages && *seen.begin() == 0, "every page handed out once");
  ok &= expect(pool.free_pages() == 0, "no free pages left");
  ok &= expect(pool.acquire() == -1, "acquire on an empty pool reports -1");

  for (int page : seen) ok &= expect(pool.release(page), "release of a held page");
  ok &= expect(pool.free_pages() == kPages && pool.bytes_in_use() == 0, "all pages returned");
  ok &= expect(pool.high_water_mark() == kPageBytes * kPages, "high water mark survives release");
  return ok;
}

bool test_bad_release() {
  llm::PagePool pool;
  pool.allocate(kPageBytes, kPages);
  const int page = pool.acquire();

  bool ok = expect(pool.release(page), "first release");
  ok &= expect(!pool.release(page), "double release rejected");
  ok &= expect(!pool.release(-1), "negative id rejected");
  ok &= expect(!pool.release(kPages), "id past the end rejected");
  ok &= expect(!pool.release(kPages - 1), "never-acquired page rejected");
  ok &= expect(pool.free_pages() == kPages, "rejected releases change nothing");
  return ok;
}

// Random traffic, checked against a plain host-side copy of what should be held.
bool test_stress() {
  llm::PagePool pool;
  pool.allocate(kPageBytes, kPages);

  std::mt19937 rng(1234);
  std::vector<int> held;
  std::size_t peak = 0;

  for (int step = 0; step < kStressSteps; ++step) {
    if (rng() % 2 == 0) {
      const int page = pool.acquire();
      if (static_cast<int>(held.size()) == kPages) {
        if (page != -1) return expect(false, "acquire succeeded on a full pool");
      } else {
        for (int other : held) {
          if (other == page) return expect(false, "page handed out twice");
        }
        if (page < 0) return expect(false, "acquire failed with pages free");
        held.push_back(page);
      }
    } else if (!held.empty()) {
      const std::size_t slot = rng() % held.size();
      if (!pool.release(held[slot])) return expect(false, "release of a held page");
      held[slot] = held.back();
      held.pop_back();
    }

    peak = std::max(peak, held.size() * kPageBytes);
    if (pool.bytes_in_use() != held.size() * kPageBytes) {
      return expect(false, "bytes_in_use disagrees with held pages");
    }
    if (pool.free_pages() + static_cast<int>(held.size()) != kPages) {
      return expect(false, "pages lost or duplicated");
    }
  }
  return expect(pool.high_water_mark() == peak, "high water mark matches the true peak");
}

// A page bleeding into its neighbour would overwrite the neighbour's pattern.
bool test_pages_do_not_overlap() {
  llm::PagePool pool;
  pool.allocate(kPageBytes, kPages);

  std::vector<float> host(kFloatsPerPage);
  for (int page = 0; page < kPages; ++page) {
    for (int i = 0; i < kFloatsPerPage; ++i) host[i] = page * 1000.0f + i;
    LLM_CUDA_CHECK(
        cudaMemcpy(pool.page_data(page), host.data(), kPageBytes, cudaMemcpyHostToDevice));
  }

  for (int page = 0; page < kPages; ++page) {
    LLM_CUDA_CHECK(
        cudaMemcpy(host.data(), pool.page_data(page), kPageBytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < kFloatsPerPage; ++i) {
      if (host[i] != page * 1000.0f + i) return expect(false, "page contents were overwritten");
    }
  }
  return true;
}

}  // namespace

int main() {
  bool ok = test_exhaust_and_refill();
  ok &= test_bad_release();
  ok &= test_stress();
  ok &= test_pages_do_not_overlap();

  if (!ok) return EXIT_FAILURE;
  std::printf("PASS: PagePool bookkeeping held over %d random steps, %d pages stayed separate\n",
              kStressSteps, kPages);
  return EXIT_SUCCESS;
}
