#───────────────────────────────────────────────────────────────────────────────
# ROCKS: Rao-blackwellized Online Conjugate KSC-Storvik
# Robust real-time stochastic volatility tracking for high-frequency trading
#───────────────────────────────────────────────────────────────────────────────

# Options
option(RBPF_USE_DOUBLE "Use double precision (slower but more accurate)" OFF)
option(RBPF_BUILD_BENCH "Build benchmark executable" ON)
option(RBPF_BUILD_TESTS "Build test executables" ON)
option(RBPF_ENABLE_STORVIK "Enable Sleeping Storvik parameter learning" ON)

#───────────────────────────────────────────────────────────────────────────────
# ROCKS Core Library
#───────────────────────────────────────────────────────────────────────────────

set(ROCKS_SOURCES
    # Core Rao-Blackwellized particle filter
    rao-blackwellized/rbpf_ksc.c
    
    # Auxiliary particle filter
    apf/rbpf_apf.c
)

# Add Storvik parameter learning and related modules if enabled
if(RBPF_ENABLE_STORVIK)
    list(APPEND ROCKS_SOURCES
        # Storvik online parameter learning
        storvik/rbpf_param_learn.c
        storvik/rbpf_adaptive_forgetting.c
        
        # Robust OCSN likelihood
        ocsn/rbpf_ocsn_robust.c
        
        # Integration layer
        rbpf_ksc_param_integration.c
        
        # Tuner
        tuner/rbpf_tuner.c
        
        # Preprocessor
        preprocess/rbpf_preprocess.c
        
        # Watchdog (not yet integrated)
        # watchdog/rbpf_watchdog.c
        
        # Hawkes intensity
        hawkes/hawkes_intensity.c
        
        # MMPF (Multi-Model Particle Filter)
        mmpf/mmpf_rocks.c
    )
endif()

add_library(rocks STATIC ${ROCKS_SOURCES})

target_include_directories(rocks 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/rao-blackwellized
        ${CMAKE_CURRENT_SOURCE_DIR}/apf
        ${CMAKE_CURRENT_SOURCE_DIR}/storvik
        ${CMAKE_CURRENT_SOURCE_DIR}/ocsn
        ${CMAKE_CURRENT_SOURCE_DIR}/tuner
        ${CMAKE_CURRENT_SOURCE_DIR}/preprocess
        # ${CMAKE_CURRENT_SOURCE_DIR}/watchdog  # Not yet integrated
        ${CMAKE_CURRENT_SOURCE_DIR}/hawkes
        ${CMAKE_CURRENT_SOURCE_DIR}/pmmh
        ${CMAKE_CURRENT_SOURCE_DIR}/duration_tracker
        ${CMAKE_CURRENT_SOURCE_DIR}/mmpf
)

target_link_libraries(rocks 
    PUBLIC 
        MKL::MKL
)

target_compile_options(rocks PRIVATE ${COMMON_FLAGS})
target_compile_definitions(rocks PRIVATE ${COMMON_DEFS})

if(RBPF_USE_DOUBLE)
    target_compile_definitions(rocks PUBLIC RBPF_USE_DOUBLE=1)
endif()

if(RBPF_ENABLE_STORVIK)
    target_compile_definitions(rocks PUBLIC RBPF_ENABLE_STORVIK=1)
endif()

# Alias for backward compatibility
add_library(rbpf_ksc ALIAS rocks)

#───────────────────────────────────────────────────────────────────────────────
# Test Executables
#───────────────────────────────────────────────────────────────────────────────

if(RBPF_BUILD_TESTS)
    # Integration test (Storvik + RBPF-KSC)
    if(RBPF_ENABLE_STORVIK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_rbpf_integration.c)
        add_executable(test_rbpf_integration
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_rbpf_integration.c
        )
        target_link_libraries(test_rbpf_integration PRIVATE rocks)
        target_compile_options(test_rbpf_integration PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_rbpf_integration PRIVATE ${COMMON_DEFS})
    endif()

    # Comprehensive comparison test v2
    if(RBPF_ENABLE_STORVIK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_rbpf_comparison.c)
        add_executable(test_rbpf_comparison
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_rbpf_comparison.c
        )
        target_link_libraries(test_rbpf_comparison PRIVATE rocks)
        target_compile_options(test_rbpf_comparison PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_rbpf_comparison PRIVATE ${COMMON_DEFS})
    endif()

    # Legacy comparison test
    if(RBPF_ENABLE_STORVIK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/rbpf_comparison_test.c)
        add_executable(rbpf_comparison_test
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/rbpf_comparison_test.c
        )
        target_link_libraries(rbpf_comparison_test PRIVATE rocks)
        target_compile_options(rbpf_comparison_test PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(rbpf_comparison_test PRIVATE ${COMMON_DEFS})
    endif()

    # Param learning standalone test
    if(RBPF_ENABLE_STORVIK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_param_learn.c)
        add_executable(test_param_learn
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_param_learn.c
        )
        target_link_libraries(test_param_learn PRIVATE rocks)
        target_compile_options(test_param_learn PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_param_learn PRIVATE ${COMMON_DEFS})
    endif()

    # RBPF Parameter Tuner test
    if(RBPF_ENABLE_STORVIK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_rbpf_tuner.c)
        add_executable(test_rbpf_tuner
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_rbpf_tuner.c
        )
        target_link_libraries(test_rbpf_tuner PRIVATE rocks)
        target_compile_options(test_rbpf_tuner PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_rbpf_tuner PRIVATE ${COMMON_DEFS})
    endif()

    # Realistic scenario tests
    if(RBPF_ENABLE_STORVIK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_rbpf_scenarios.c)
        add_executable(test_rbpf_scenarios
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_rbpf_scenarios.c
        )
        target_link_libraries(test_rbpf_scenarios PRIVATE rocks)
        target_compile_options(test_rbpf_scenarios PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_rbpf_scenarios PRIVATE ${COMMON_DEFS})
    endif()

    # CSV output test for Jupyter visualization
    if(RBPF_ENABLE_STORVIK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_rbpf_csv.c)
        add_executable(test_rbpf_csv
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_rbpf_csv.c
        )
        target_link_libraries(test_rbpf_csv PRIVATE rocks)
        target_compile_options(test_rbpf_csv PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_rbpf_csv PRIVATE ${COMMON_DEFS})
    endif()

    # MMPF correctness tests
    if(RBPF_ENABLE_STORVIK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_mmpf_correctness.c)
        add_executable(test_mmpf_correctness
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/test_mmpf_correctness.c
        )
        target_link_libraries(test_mmpf_correctness PRIVATE rocks)
        target_compile_options(test_mmpf_correctness PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_mmpf_correctness PRIVATE ${COMMON_DEFS})
    endif()

    # MMPF real data integration test
    if(RBPF_ENABLE_STORVIK)
        add_executable(test_mmpf_real_data
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/mmpf_real_data_test/test_mmpf_real_data.c
        )
        target_link_libraries(test_mmpf_real_data PRIVATE rocks)
        target_compile_options(test_mmpf_real_data PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(test_mmpf_real_data PRIVATE ${COMMON_DEFS})
    endif()

    # Example usage
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/example_usage.c)
        add_executable(example_usage
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/example_usage.c
        )
        target_link_libraries(example_usage PRIVATE rocks)
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
        target_link_libraries(rbpf_ksc_bench PRIVATE rocks)
        target_compile_options(rbpf_ksc_bench PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(rbpf_ksc_bench PRIVATE ${COMMON_DEFS})
    endif()

    # PMMH benchmark
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/bench_pmmh_mkl.c)
        add_executable(bench_pmmh_mkl
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/bench_pmmh_mkl.c
        )
        target_link_libraries(bench_pmmh_mkl PRIVATE rocks)
        target_compile_options(bench_pmmh_mkl PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(bench_pmmh_mkl PRIVATE ${COMMON_DEFS})
    endif()

    # PMMH tuning
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../test/tune_pmmh.c)
        add_executable(tune_pmmh
            ${CMAKE_CURRENT_SOURCE_DIR}/../test/tune_pmmh.c
        )
        target_link_libraries(tune_pmmh PRIVATE rocks)
        target_compile_options(tune_pmmh PRIVATE ${COMMON_FLAGS})
        target_compile_definitions(tune_pmmh PRIVATE ${COMMON_DEFS})
    endif()
endif()

#───────────────────────────────────────────────────────────────────────────────
# Installation
#───────────────────────────────────────────────────────────────────────────────

install(TARGETS rocks
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# Core headers
set(ROCKS_HEADERS
    mkl_config.h
    rao-blackwellized/rbpf_ksc.h
)

# Storvik and related headers
if(RBPF_ENABLE_STORVIK)
    list(APPEND ROCKS_HEADERS
        rbpf_ksc_param_integration.h
        storvik/rbpf_param_learn.h
        tuner/rbpf_tuner.h
        preprocess/rbpf_preprocess.h
        # watchdog/rbpf_watchdog.h  # Not yet integrated
        hawkes/hawkes_intensity.h
        pmmh/pmmh_mkl.h
        duration_tracker/duration_tracker_fast.h
        mmpf/mmpf_rocks.h
    )
endif()

install(FILES ${ROCKS_HEADERS}
    DESTINATION include/rocks
)

if(MSVC)
    target_compile_options(rocks PRIVATE /openmp:experimental)
endif()