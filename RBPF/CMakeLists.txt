#───────────────────────────────────────────────────────────────────────────────
# 3-RBPF: Rao-Blackwellized Particle Filter with Kim-Shephard-Chib observation
#
# Different approach from PF2D:
#   - Continuous state (log-vol) handled by Kalman filter (exact)
#   - Only discrete state (regime) is sampled
#   - KSC 1998 mixture approximation for log-chi-squared observation noise
#   - Much fewer particles needed (200 vs 2000+)
#
# Usage from root:
#   add_subdirectory(3-RBPF)
#───────────────────────────────────────────────────────────────────────────────

#───────────────────────────────────────────────────────────────────────────────
# Options
#───────────────────────────────────────────────────────────────────────────────

option(RBPF_BUILD_BENCH "Build RBPF benchmark executable" ON)
option(RBPF_USE_DOUBLE "Use double precision (default OFF for latency)" OFF)

#───────────────────────────────────────────────────────────────────────────────
# Static Library
#───────────────────────────────────────────────────────────────────────────────

add_library(rbpf_ksc STATIC
    rbpf_ksc.c
    rbpf_apf.c
    rbpf_pipeline.c
)

target_include_directories(rbpf_ksc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(rbpf_ksc PUBLIC
    MKL::MKL
    $<$<NOT:$<BOOL:${WIN32}>>:OpenMP::OpenMP_C>
)

target_compile_options(rbpf_ksc PRIVATE ${COMMON_FLAGS})
if(DEFINED COMMON_DEFS)
    target_compile_definitions(rbpf_ksc PRIVATE ${COMMON_DEFS})
endif()

if(RBPF_USE_DOUBLE)
    target_compile_definitions(rbpf_ksc PUBLIC RBPF_USE_DOUBLE)
endif()

#───────────────────────────────────────────────────────────────────────────────
# Shared Library
#───────────────────────────────────────────────────────────────────────────────

add_library(rbpf_ksc_shared SHARED
    rbpf_ksc.c
    rbpf_apf.c
    rbpf_pipeline.c
)

target_include_directories(rbpf_ksc_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(rbpf_ksc_shared PUBLIC
    MKL::MKL
    $<$<NOT:$<BOOL:${WIN32}>>:OpenMP::OpenMP_C>
)

target_compile_options(rbpf_ksc_shared PRIVATE ${COMMON_FLAGS})
if(DEFINED COMMON_DEFS)
    target_compile_definitions(rbpf_ksc_shared PRIVATE ${COMMON_DEFS})
endif()

if(RBPF_USE_DOUBLE)
    target_compile_definitions(rbpf_ksc_shared PUBLIC RBPF_USE_DOUBLE)
endif()

set_target_properties(rbpf_ksc_shared PROPERTIES
    OUTPUT_NAME "rbpf_ksc"
)

if(WIN32)
    # Windows: export all symbols or use .def file
    set_target_properties(rbpf_ksc_shared PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
else()
    set_target_properties(rbpf_ksc_shared PROPERTIES PREFIX "")
endif()

#───────────────────────────────────────────────────────────────────────────────
# Benchmark (in test/ folder, but linked here)
#───────────────────────────────────────────────────────────────────────────────

if(RBPF_BUILD_BENCH)
    add_executable(bench_rbpf_ksc
        ${CMAKE_SOURCE_DIR}/test/rbpf_ksc_bench.c
    )

    target_include_directories(bench_rbpf_ksc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(bench_rbpf_ksc PRIVATE
        rbpf_ksc
    )

    target_compile_options(bench_rbpf_ksc PRIVATE ${COMMON_FLAGS})
    if(DEFINED COMMON_DEFS)
        target_compile_definitions(bench_rbpf_ksc PRIVATE ${COMMON_DEFS})
    endif()

    if(UNIX)
        target_link_libraries(bench_rbpf_ksc PRIVATE m)
    endif()

    # Register as CTest
    add_test(NAME rbpf_ksc_benchmark COMMAND bench_rbpf_ksc)
endif()

#───────────────────────────────────────────────────────────────────────────────
# Scenario Tests (realistic market scenarios)
#───────────────────────────────────────────────────────────────────────────────

option(RBPF_BUILD_TESTS "Build RBPF scenario tests" ON)

if(RBPF_BUILD_TESTS)
    add_executable(test_rbpf_scenarios
        ${CMAKE_SOURCE_DIR}/test/test_rbpf_scenarios.c
    )

    target_include_directories(test_rbpf_scenarios PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(test_rbpf_scenarios PRIVATE
        rbpf_ksc
    )

    target_compile_options(test_rbpf_scenarios PRIVATE ${COMMON_FLAGS})
    if(DEFINED COMMON_DEFS)
        target_compile_definitions(test_rbpf_scenarios PRIVATE ${COMMON_DEFS})
    endif()

    if(UNIX)
        target_link_libraries(test_rbpf_scenarios PRIVATE m)
    endif()

    # Register as CTest
    add_test(NAME rbpf_scenario_tests COMMAND test_rbpf_scenarios)
endif()

#───────────────────────────────────────────────────────────────────────────────
# Install
#───────────────────────────────────────────────────────────────────────────────

install(TARGETS rbpf_ksc rbpf_ksc_shared
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES rbpf_ksc.h
    DESTINATION include/particle_filter
)

#───────────────────────────────────────────────────────────────────────────────
# Summary
#───────────────────────────────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "=== RBPF-KSC Configuration ===")
message(STATUS "Precision:        ${RBPF_USE_DOUBLE} (OFF=float, ON=double)")
message(STATUS "Build benchmark:  ${RBPF_BUILD_BENCH}")
message(STATUS "Build tests:      ${RBPF_BUILD_TESTS}")
message(STATUS "")