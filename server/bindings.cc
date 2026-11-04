// Python module `llm_engine_py`: the C++ AsyncEngine, called without holding the GIL.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "llm/async_engine.h"

namespace py = pybind11;

namespace {

using Event = std::tuple<std::int64_t, std::int32_t, int, std::string>;  // id, token, index, finish

std::unique_ptr<llm::AsyncEngine> make_engine(const std::string& model_path, int max_tokens,
                                              int max_seqs, int kv_pages, int max_waiting) {
  llm::EngineOptions options;
  options.model_path = model_path;
  options.max_tokens = max_tokens;
  options.max_seqs = max_seqs;
  options.kv_pages = kv_pages;
  options.max_waiting = max_waiting;
  py::gil_scoped_release release;
  return std::make_unique<llm::AsyncEngine>(options);
}

std::vector<Event> poll(llm::AsyncEngine& engine, int timeout_ms) {
  std::vector<llm::TokenEvent> events;
  {
    py::gil_scoped_release release;
    events = engine.poll(timeout_ms);
  }
  std::vector<Event> out;
  out.reserve(events.size());
  for (const llm::TokenEvent& e : events) {
    out.emplace_back(e.id, e.token, e.index, llm::finish_reason_name(e.finish));
  }
  return out;
}

// Returns (status, message): status is "ok", "invalid", "busy" or "unavailable".
std::tuple<std::string, std::string> submit(llm::AsyncEngine& engine, std::int64_t id,
                                            std::vector<std::int32_t> prompt, int max_new_tokens,
                                            std::vector<std::int32_t> stop_ids, float temperature,
                                            float top_p, int top_k, std::int64_t seed, int timeout_ms) {
  llm::Request r;
  r.id = id;
  r.prompt = std::move(prompt);
  r.max_new_tokens = max_new_tokens;
  r.stop_ids = std::move(stop_ids);
  r.sampling.temperature = temperature;
  r.sampling.top_p = top_p;
  r.sampling.top_k = top_k;
  r.sampling.seed = seed;
  r.timeout_ms = timeout_ms;
  llm::SubmitResult result;
  {
    py::gil_scoped_release release;
    result = engine.submit(std::move(r));
  }
  return {llm::submit_status_name(result.status), result.message};
}

}  // namespace

PYBIND11_MODULE(llm_engine_py, m) {
  m.doc() = "bare-cuda-4gib engine: continuous batching on a worker thread";

  py::class_<llm::AsyncEngine>(m, "Engine")
      .def(py::init(&make_engine), py::arg("model_path"), py::arg("max_tokens") = 1024,
           py::arg("max_seqs") = 8, py::arg("kv_pages") = 0, py::arg("max_waiting") = 64)
      .def("submit", &submit, py::arg("id"), py::arg("prompt"), py::arg("max_new_tokens"),
           py::arg("stop_ids") = std::vector<std::int32_t>{}, py::arg("temperature") = 0.0f,
           py::arg("top_p") = 1.0f, py::arg("top_k") = 0, py::arg("seed") = -1,
           py::arg("timeout_ms") = 0,
           "Queues a request; returns (status, message). temperature 0 is greedy.")
      .def("cancel", &llm::AsyncEngine::cancel, py::arg("id"),
           py::call_guard<py::gil_scoped_release>())
      .def("poll", &poll, py::arg("timeout_ms") = 100,
           "Waits for events: a list of (id, token, index, finish) tuples.")
      .def("stop", &llm::AsyncEngine::stop, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("healthy", &llm::AsyncEngine::healthy)
      .def_property_readonly("failure", &llm::AsyncEngine::failure)
      .def("_fail_for_test", &llm::AsyncEngine::fail_for_test, py::arg("message"))
      .def_property_readonly("max_tokens", [](const llm::AsyncEngine& e) { return e.limits().max_tokens; })
      .def_property_readonly("max_seqs", [](const llm::AsyncEngine& e) { return e.limits().max_seqs; })
      .def_property_readonly("kv_pages", [](const llm::AsyncEngine& e) { return e.limits().kv_pages; })
      .def_property_readonly("vocab_size", [](const llm::AsyncEngine& e) { return e.config().vocab_size; });
}
