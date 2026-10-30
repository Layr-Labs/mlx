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
  add_executable(metal_fence_chain metal_fence_chain.cpp)
  target_link_libraries(metal_fence_chain PRIVATE mlx)
  foreach(ops IN ITEMS 8 50 1024)
    add_test(NAME metal_fence_chain_${ops} COMMAND metal_fence_chain)
    set_tests_properties(
      metal_fence_chain_${ops}
      PROPERTIES TIMEOUT 60 ENVIRONMENT
                 "MLX_METAL_FAST_SYNCH=1;MLX_MAX_OPS_PER_BUFFER=${ops}")
  endforeach()
  list(APPEND GPU_TEST_SOURCES residency_tests.cpp gptoss_mxfp4_tests.cpp
       quantized_tile_tests.cpp qmv_batch_invariant_tests.cpp)
  # MLX reads MLX_QMV_BATCH_INVARIANT once in each process, so this entry runs
  # the case again in a new process with the switch on. The tests entries run it
  # with the switch off. The entry fails also if no test case has this name.
  add_test(
    NAME qmv_batch_invariant
    COMMAND
      tests
      "--test-case=quantized matmul rows keep their bits below the vector limit"
  )
  set_tests_properties(
    qmv_batch_invariant
    PROPERTIES ENVIRONMENT "MLX_QMV_BATCH_INVARIANT=1" PASS_REGULAR_EXPRESSION
               "test cases: +1 \\| +1 passed \\| +0 failed")
endif()

if(TARGET jaccl)
  target_sources(tests PRIVATE jaccl_tests.cpp)
  target_include_directories(
    tests PRIVATE ${PROJECT_SOURCE_DIR}/mlx/distributed/jaccl/lib)
  get_target_property(mlx_link_libraries mlx LINK_LIBRARIES)
  if(BUILD_SHARED_LIBS OR NOT jaccl IN_LIST mlx_link_libraries)
    target_link_libraries(tests PRIVATE jaccl)
  endif()
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
          config_tests.cpp
          custom_vjp_tests.cpp
          creations_tests.cpp
          device_tests.cpp
          distributed_sealed_stream_tests.cpp
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
    # The registration cases exit with 77 when the device has no residency sets,
    # so ctest reports them as skipped.
    if(case MATCHES "^registration-")
      set_tests_properties(process_allocation_cap_metal.${case}
                           PROPERTIES SKIP_RETURN_CODE 77)
    endif()
  endforeach()
endif()
