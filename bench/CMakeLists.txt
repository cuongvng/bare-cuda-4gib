add_executable(llm-devinfo src/devinfo_main.cc)
target_link_libraries(llm-devinfo PRIVATE llm_engine)

add_executable(llm-infer src/main.cc)
target_link_libraries(llm-infer PRIVATE llm_engine)

add_executable(llm-bench src/bench_main.cc)
target_link_libraries(llm-bench PRIVATE llm_engine)

add_executable(llm-ppl src/ppl_main.cc)
target_link_libraries(llm-ppl PRIVATE llm_engine)

add_executable(llm-batch-check src/batch_check_main.cc)
target_link_libraries(llm-batch-check PRIVATE llm_engine)
