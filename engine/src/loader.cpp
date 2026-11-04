#include "llm/loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "llm/error.h"

namespace llm {
namespace {

// Little-endian fixed-width reads from a byte cursor. 
class Cursor {
 public:
  Cursor(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

  template <typename T>
  T read() {
    require(sizeof(T));
    T value;
    std::memcpy(&value, data_ + pos_, sizeof(T));
    pos_ += sizeof(T);
    return value;
  }

  std::string read_string(std::uint32_t len) {
    require(len);
    std::string s(reinterpret_cast<const char*>(data_ + pos_), len);
    pos_ += len;
    return s;
  }

  std::size_t pos() const { return pos_; }

 private:
  void require(std::size_t n) {
    if (pos_ + n > size_) {
      raise_error("[llm] loader: truncated file (need " + std::to_string(n) + " bytes at offset " +
                  std::to_string(pos_) + ", have " + std::to_string(size_) + ")");
    }
  }

  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t pos_ = 0;
};

[[noreturn]] void fail(const std::string& path, const std::string& reason) {
  raise_error("[llm] loader: " + path + ": " + reason);
}

}  // namespace

ModelFile ModelFile::load(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) fail(path, "cannot open file");

  file.seekg(0, std::ios::end);
  const std::streamoff size = file.tellg();
  if (size < 0) fail(path, "cannot determine file size");
  file.seekg(0, std::ios::beg);

  ModelFile out;
  out.storage_.resize(static_cast<std::size_t>(size));
  if (size > 0 && !file.read(reinterpret_cast<char*>(out.storage_.data()), size)) {
    fail(path, "read failed");
  }

  Cursor cur(out.storage_.data(), out.storage_.size());

  char magic[4];
  for (char& c : magic) c = static_cast<char>(cur.read<std::uint8_t>());
  if (std::memcmp(magic, "LLM1", 4) != 0) fail(path, "bad magic (not an LLM1 file)");

  const std::uint32_t version = cur.read<std::uint32_t>();
  if (version != 1) fail(path, "unsupported format version " + std::to_string(version));

  ModelConfig& cfg = out.config_;
  cfg.vocab_size = cur.read<std::uint32_t>();
  cfg.hidden_size = cur.read<std::uint32_t>();
  cfg.num_layers = cur.read<std::uint32_t>();
  cfg.num_heads = cur.read<std::uint32_t>();
  cfg.num_kv_heads = cur.read<std::uint32_t>();
  cfg.head_dim = cur.read<std::uint32_t>();
  cfg.intermediate_size = cur.read<std::uint32_t>();
  cfg.max_seq_len = cur.read<std::uint32_t>();
  cfg.rope_theta = cur.read<float>();
  cfg.rms_norm_eps = cur.read<float>();

  const std::uint32_t num_tensors = cur.read<std::uint32_t>();

  struct DirEntry {
    std::string name;
    DType dtype;
    std::array<std::uint32_t, kMaxTensorDims> shape{};
    int ndim;
    std::uint64_t offset;
    std::uint64_t nbytes;
  };
  std::vector<DirEntry> entries;
  entries.reserve(num_tensors);

  for (std::uint32_t i = 0; i < num_tensors; ++i) {
    DirEntry e;
    const std::uint32_t name_len = cur.read<std::uint32_t>();
    e.name = cur.read_string(name_len);
    e.dtype = static_cast<DType>(cur.read<std::uint32_t>());
    const std::uint32_t ndim = cur.read<std::uint32_t>();
    if (ndim > kMaxTensorDims) fail(path, "tensor '" + e.name + "' has ndim > kMaxTensorDims");
    e.ndim = static_cast<int>(ndim);
    for (std::uint32_t d = 0; d < ndim; ++d) e.shape[d] = cur.read<std::uint32_t>();
    e.offset = cur.read<std::uint64_t>();
    e.nbytes = cur.read<std::uint64_t>();
    entries.push_back(std::move(e));
  }

  const std::size_t data_section_start = cur.pos();
  for (const DirEntry& e : entries) {
    const std::size_t begin = data_section_start + static_cast<std::size_t>(e.offset);
    const std::size_t end = begin + static_cast<std::size_t>(e.nbytes);
    if (end > out.storage_.size()) {
      fail(path, "tensor '" + e.name + "' data range exceeds file size");
    }
    TensorView view;
    view.data = out.storage_.data() + begin;
    view.dtype = e.dtype;
    view.ndim = e.ndim;
    view.shape = e.shape;
    view.nbytes = e.nbytes;
    out.tensors_.emplace(e.name, view);
  }

  return out;
}

const TensorView* ModelFile::find(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const TensorView& ModelFile::require(const std::string& name) const {
  const TensorView* v = find(name);
  if (!v) {
    raise_error("[llm] loader: required tensor '" + name + "' not found in model file");
  }
  return *v;
}

}  // namespace llm
