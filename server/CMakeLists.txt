# Python module for the HTTP server; built only when pybind11 is installed.
#   pip install -r server/requirements.txt
#   cmake -B build -Dpybind11_DIR=$(python -m pybind11 --cmakedir) -DPython3_EXECUTABLE=$(which python)
find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND OR NOT Python3_Development.Module_FOUND)
  message(STATUS "llm_engine_py skipped: pybind11 or Python headers not found (pass -Dpybind11_DIR=...)")
  return()
endif()

pybind11_add_module(llm_engine_py bindings.cc)
target_link_libraries(llm_engine_py PRIVATE llm_engine)
# Lands next to the other build outputs, where server/app.py looks by default.
set_target_properties(llm_engine_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)
message(STATUS "llm_engine_py: Python ${Python3_VERSION} at ${Python3_EXECUTABLE}")
