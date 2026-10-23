cmake_minimum_required(VERSION 3.24)
project(jaccl_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(JACCL_TESTS_TSAN "Use ThreadSanitizer in the JACCL tests" OFF)

if(JACCL_TESTS_TSAN)
  set(SANITIZE -fsanitize=thread)
else()
  set(SANITIZE -fsanitize=address,undefined)
endif()

enable_testing()

set(JACCL_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../../mlx/distributed/jaccl/lib)

# The tests use the JACCL headers and the fake verbs, not the jaccl library.
function(jaccl_test name)
  add_executable(${name} ${name}.cpp fake_rdma.cpp)
  target_include_directories(${name} PRIVATE fake_verbs ${JACCL_LIB})
  target_compile_options(${name} PRIVATE -O1 -g ${SANITIZE})
  target_link_options(${name} PRIVATE ${SANITIZE})
endfunction()

jaccl_test(send_frame_tests)
jaccl_test(progress_guard_tests)
jaccl_test(startup_wait_budget_tests)
jaccl_test(startup_wait_budget_group_tests)
jaccl_test(startup_wait_budget_api_tests)
target_include_directories(startup_wait_budget_api_tests PRIVATE ../..)
jaccl_test(exchange_tests)
target_include_directories(exchange_tests PRIVATE ../..)
add_test(NAME exchange COMMAND exchange_tests)
set_tests_properties(exchange PROPERTIES ENVIRONMENT
                                         "JACCL_TIMEOUT_ACTION=throw")
add_test(NAME startup_wait_budget_api COMMAND startup_wait_budget_api_tests)
add_test(NAME startup_wait_budget_groups
         COMMAND startup_wait_budget_group_tests)
set_tests_properties(startup_wait_budget_groups
                     PROPERTIES ENVIRONMENT "JACCL_TIMEOUT_ACTION=throw")
add_test(NAME startup_wait_budget COMMAND startup_wait_budget_tests)

add_test(NAME send_frame COMMAND send_frame_tests)

# The guard reads its limit and action once in a process, so each mode is one
# test. teardown-exit and default-exit start this program again as children.
foreach(
  mode
  checks
  disabled
  alias
  precedence
  negative
  not-a-number
  default
  teardown-exit
  default-exit)
  add_test(NAME progress_guard_${mode} COMMAND progress_guard_tests ${mode})
endforeach()
