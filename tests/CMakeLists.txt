FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/onqtam/doctest.git
  GIT_TAG v2.4.12)
FetchContent_MakeAvailable(doctest)

add_executable(tests ${PROJECT_SOURCE_DIR}/tests/tests.cpp)

if(MLX_BUILD_METAL OR MLX_BUILD_CUDA)
  set(GPU_TEST_SOURCES gpu_tests.cpp)
endif()

if(MLX_BUILD_CUDA)
  list(APPEND GPU_TEST_SOURCES cuda_tests.cpp)
  target_link_libraries(tests PRIVATE CUDA::cudart)
endif()

if(MLX_BUILD_METAL)
  list(APPEND GPU_TEST_SOURCES residency_tests.cpp gptoss_mxfp4_tests.cpp)
endif()

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

target_sources(
  tests
  PRIVATE allocator_tests.cpp
          array_tests.cpp
          arg_reduce_tests.cpp
          autograd_tests.cpp
          blas_tests.cpp
          compile_tests.cpp
          custom_vjp_tests.cpp
          creations_tests.cpp
          device_tests.cpp
          einsum_tests.cpp
          export_import_tests.cpp
          eval_tests.cpp
          fft_tests.cpp
          load_tests.cpp
          ops_tests.cpp
          random_tests.cpp
          scheduler_tests.cpp
          utils_tests.cpp
          vmap_tests.cpp
          linalg_tests.cpp
          ${GPU_TEST_SOURCES})

target_link_libraries(tests PRIVATE mlx doctest)
target_compile_options(tests PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(tests)
add_test(NAME tests COMMAND tests)

# The standalone tests of the process allocation cap. The cap is install-once in
# each process, so these tests are not in the tests binary. Each ctest entry
# runs in a fresh process, and the Metal test runs one case in each process.
if(MLX_BUILD_METAL)
  add_executable(
    process_allocation_cap
    ${PROJECT_SOURCE_DIR}/tests/standalone/process_allocation_cap.cpp)
  target_link_libraries(process_allocation_cap PRIVATE mlx)
  # The test checks its results with assert().
  target_compile_options(process_allocation_cap PRIVATE -UNDEBUG)
  add_test(NAME process_allocation_cap COMMAND process_allocation_cap)

  add_executable(
    process_allocation_cap_metal
    ${PROJECT_SOURCE_DIR}/tests/standalone/process_allocation_cap_metal.cpp)
  # The test replaces Metal device methods with the Objective-C runtime.
  target_link_libraries(process_allocation_cap_metal PRIVATE mlx objc)
  foreach(
    case
    install
    retained
    churn
    foreign
    gpu
    late
    concurrent
    footprint-delta
    gpu-footprint
    nil-foreign
    registration-capped
    registration-foreign
    registration-ordinary
    registration-ordinary-foreign)
    add_test(NAME process_allocation_cap_metal.${case}
             COMMAND process_allocation_cap_metal ${case})
  endforeach()
endif()
