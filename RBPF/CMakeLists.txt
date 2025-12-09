#───────────────────────────────────────────────────────────────────────────────
# RBPF Library and Tests
#───────────────────────────────────────────────────────────────────────────────

# Options
option(RBPF_USE_DOUBLE "Use double precision (slower but more accurate)" OFF)
option(RBPF_BUILD_BENCH "Build benchmark executable" ON)
option(RBPF_BUILD_TESTS "Build test executables" ON)
option(RBPF_ENABLE_STORVIK "Enable Sleeping Storvik parameter learning" ON)

#───────────────────────────────────────────────────────────────────────────────
# RBPF Core Library
#───────────────────────────────────────────────────────────────────────────────

set(RBPF_SOURCES
    rbpf_ksc.c
    rbpf_apf.c
)

# Add Storvik parameter learning if enabled
if(RBPF_ENABLE_STORVIK)
    list(APPEND RBPF_SOURCES
        rbpf_param_learn.c
        rbpf_ksc_param_integration.c
    )
endif()

add_library(rbpf_ksc STATIC ${RBPF_SOURCES})

target_include_directories(rbpf_ksc 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(rbpf_ksc 
    PUBLIC 
        MKL::MKL
)

target_compile_options(rbpf_ksc PRIVATE ${COMMON_FLAGS})
target_compile_definitions(rbpf_ksc PRIVATE ${COMMON_DEFS})

if(RBPF_USE_DOUBLE)
    target_compile_definitions(rbpf_ksc PUBLIC RBPF_USE_DOUBLE=1)
endif()

if(RBPF_ENABLE_STORVIK)
    target_compile_definitions(rbpf_ksc PUBLIC RBPF_ENABLE_STORVIK=1)
endif()

#───────────────────────────────────────────────────────────────────────────────
# Test Executables
#───────────────────────────────────────────────────────────────────────────────

if(RBPF_BUILD_TESTS)
    # Integration test (Storvik + RBPF-KSC)
    if(RBPF_ENABLE_STORVIK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_rbpf_integration.c)
        add_executable(test_rbpf_integration
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_rbpf_integration.c
        )
        target_link_libraries(test_rbpf_integration PRIVATE rbpf_ksc)
        target_compile_options(test_rbpf_integration PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_rbpf_integration PRIVATE ${COMMON_DEFS})
    endif()

    # Comprehensive comparison test (Baseline vs Liu-West vs Storvik vs Storvik+APF)
    if(RBPF_ENABLE_STORVIK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/rbpf_comparison_test.c)
        add_executable(rbpf_comparison_test
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/rbpf_comparison_test.c
        )
        target_link_libraries(rbpf_comparison_test PRIVATE rbpf_ksc)
        target_compile_options(rbpf_comparison_test PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(rbpf_comparison_test PRIVATE ${COMMON_DEFS})
    endif()

    # Param learning standalone test (no MKL dependency in test itself)
    if(RBPF_ENABLE_STORVIK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_param_learn.c)
        add_executable(test_param_learn
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_param_learn.c
        )
        target_link_libraries(test_param_learn PRIVATE rbpf_ksc)
        target_compile_options(test_param_learn PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_param_learn PRIVATE ${COMMON_DEFS})
    endif()

    # Scenario tests
    #if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_rbpf_scenarios.c)
    #    add_executable(test_rbpf_scenarios
    #        ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_rbpf_scenarios.c
    #    )
    #    target_link_libraries(test_rbpf_scenarios PRIVATE rbpf_ksc)
    #    target_compile_options(test_rbpf_scenarios PRIVATE ${COMMON_FLAGS})
    #    target_compile_definitions(test_rbpf_scenarios PRIVATE ${COMMON_DEFS})
    #endif()

    # Example usage
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/example_usage.c)
        add_executable(example_usage
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/example_usage.c
        )
        target_link_libraries(example_usage PRIVATE rbpf_ksc)
        target_compile_options(example_usage PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(example_usage PRIVATE ${COMMON_DEFS})
    endif()
endif()

#───────────────────────────────────────────────────────────────────────────────
# Benchmark Executables
#───────────────────────────────────────────────────────────────────────────────

if(RBPF_BUILD_BENCH)
    # KSC benchmark
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/rbpf_ksc_bench.c)
        add_executable(rbpf_ksc_bench
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/rbpf_ksc_bench.c
        )
        target_link_libraries(rbpf_ksc_bench PRIVATE rbpf_ksc)
        target_compile_options(rbpf_ksc_bench PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(rbpf_ksc_bench PRIVATE ${COMMON_DEFS})
    endif()

    # PMMH benchmark
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/bench_pmmh_mkl.c)
        add_executable(bench_pmmh_mkl
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/bench_pmmh_mkl.c
        )
        target_link_libraries(bench_pmmh_mkl PRIVATE rbpf_ksc)
        target_compile_options(bench_pmmh_mkl PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(bench_pmmh_mkl PRIVATE ${COMMON_DEFS})
    endif()

    # PMMH tuning
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/tune_pmmh.c)
        add_executable(tune_pmmh
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/tune_pmmh.c
        )
        target_link_libraries(tune_pmmh PRIVATE rbpf_ksc)
        target_compile_options(tune_pmmh PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(tune_pmmh PRIVATE ${COMMON_DEFS})
    endif()
endif()

#───────────────────────────────────────────────────────────────────────────────
# Installation
#───────────────────────────────────────────────────────────────────────────────

install(TARGETS rbpf_ksc
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

set(RBPF_HEADERS
    rbpf_ksc.h
    mkl_config.h
)

if(RBPF_ENABLE_STORVIK)
    list(APPEND RBPF_HEADERS
        rbpf_param_learn.h
        rbpf_ksc_param_integration.h
    )
endif()

install(FILES ${RBPF_HEADERS}
    DESTINATION include/rbpf
)

if(MSVC)
    target_compile_options(tune_pmmh PRIVATE /openmp:experimental)
endif()
