#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>

namespace llm {

// Any engine failure: bad input files, CUDA errors, broken invariants.
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Logs to stderr and throws, so a caller like the server can catch it instead of the process exiting.
[[noreturn]] inline void raise_error(const std::string& message) {
  std::fprintf(stderr, "%s\n", message.c_str());
  throw Error(message);
}

}  // namespace llm
