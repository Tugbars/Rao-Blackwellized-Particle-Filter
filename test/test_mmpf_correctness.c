/*=============================================================================
 * MMPF Correctness Test Suite
 *
 * Tests the IMM-MMPF-ROCKS implementation for correctness before performance.
 *
 * Test Categories:
 *   P0 - Critical (must pass):
 *     - Invariants (weights sum to 1, volatility positive, etc.)
 *     - Accessor side-effects (getters don't modify state)
 *     - Parameter sync (Storvik → RBPF bridge works)
 *     - Type conversion (double → float correct)
 *
 *   P1 - Core functionality:
 *     - Hypothesis recovery (IMM keeps dead hypotheses alive)
 *     - Switching speed (weights respond to regime changes)
 *     - OCSN integration (outlier detection works)
 *
 *   P2 - Robustness:
 *     - Extreme inputs (zero, tiny, huge returns)
 *     - Numerical stability (no NaN, no overflow)
 *     - Long-run stability
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * BUILD
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   cmake --build . --config Release --target test_mmpf_correctness
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * USAGE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   ./test_mmpf_correctness [seed]
 *
 *===========================================================================*/

#include "mmpf_rocks.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/*─────────────────────────────────────────────────────────────────────────────
 * TEST FRAMEWORK
 *───────────────────────────────────────────────────────────────────────────*/

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;
static int g_current_test_failed = 0;

#define TEST_BEGIN(name) \
    do { \
        g_tests_run++; \
        g_current_test_failed = 0; \
        printf("  [TEST] %-50s ", name); \
        fflush(stdout); \
    } while(0)

#define TEST_END() \
    do { \
        if (g_current_test_failed) { \
            g_tests_failed++; \
            printf("FAILED\n"); \
        } else { \
            g_tests_passed++; \
            printf("PASSED\n"); \
        } \
    } while(0)

#define ASSERT_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            if (!g_current_test_failed) printf("\n"); \
            printf("    ASSERT FAILED: %s\n", msg); \
            printf("      at %s:%d\n", __FILE__, __LINE__); \
            g_current_test_failed = 1; \
        } \
    } while(0)

#define ASSERT_EQ_INT(a, b, msg) \
    do { \
        if ((a) != (b)) { \
            if (!g_current_test_failed) printf("\n"); \
            printf("    ASSERT FAILED: %s\n", msg); \
            printf("      expected: %d, got: %d\n", (int)(b), (int)(a)); \
            printf("      at %s:%d\n", __FILE__, __LINE__); \
            g_current_test_failed = 1; \
        } \
    } while(0)

#define ASSERT_NEAR(a, b, tol, msg) \
    do { \
        double _a = (double)(a); \
        double _b = (double)(b); \
        if (fabs(_a - _b) > (tol)) { \
            if (!g_current_test_failed) printf("\n"); \
            printf("    ASSERT FAILED: %s\n", msg); \
            printf("      expected: %.10f, got: %.10f (tol: %.10f)\n", _b, _a, (double)(tol)); \
            printf("      at %s:%d\n", __FILE__, __LINE__); \
            g_current_test_failed = 1; \
        } \
    } while(0)

#define ASSERT_GT(a, b, msg) \
    do { \
        if (!((a) > (b))) { \
            if (!g_current_test_failed) printf("\n"); \
            printf("    ASSERT FAILED: %s\n", msg); \
            printf("      expected: > %.6f, got: %.6f\n", (double)(b), (double)(a)); \
            printf("      at %s:%d\n", __FILE__, __LINE__); \
            g_current_test_failed = 1; \
        } \
    } while(0)

#define ASSERT_GE(a, b, msg) \
    do { \
        if (!((a) >= (b))) { \
            if (!g_current_test_failed) printf("\n"); \
            printf("    ASSERT FAILED: %s\n", msg); \
            printf("      expected: >= %.6f, got: %.6f\n", (double)(b), (double)(a)); \
            printf("      at %s:%d\n", __FILE__, __LINE__); \
            g_current_test_failed = 1; \
        } \
    } while(0)

#define ASSERT_LE(a, b, msg) \
    do { \
        if (!((a) <= (b))) { \
            if (!g_current_test_failed) printf("\n"); \
            printf("    ASSERT FAILED: %s\n", msg); \
            printf("      expected: <= %.6f, got: %.6f\n", (double)(b), (double)(a)); \
            printf("      at %s:%d\n", __FILE__, __LINE__); \
            g_current_test_failed = 1; \
        } \
    } while(0)

#define ASSERT_FINITE(x, msg) \
    do { \
        if (!isfinite(x)) { \
            if (!g_current_test_failed) printf("\n"); \
            printf("    ASSERT FAILED: %s\n", msg); \
            printf("      value is not finite: %.6f\n", (double)(x)); \
            printf("      at %s:%d\n", __FILE__, __LINE__); \
            g_current_test_failed = 1; \
        } \
    } while(0)

/*─────────────────────────────────────────────────────────────────────────────
 * PCG32 RNG
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct {
    uint64_t state;
    uint64_t inc;
} pcg32_t;

static uint32_t pcg32_random(pcg32_t *rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static double pcg32_double(pcg32_t *rng) {
    return (double)pcg32_random(rng) / 4294967296.0;
}

static double pcg32_gaussian(pcg32_t *rng) {
    double u1 = pcg32_double(rng);
    double u2 = pcg32_double(rng);
    if (u1 < 1e-10) u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
}

static pcg32_t g_rng;

static void rng_init(int seed) {
    g_rng.state = seed * 12345ULL + 1;
    g_rng.inc = seed * 67890ULL | 1;
}

/* Generate returns for different regimes */
static rbpf_real_t gen_calm_return(void) {
    /* σ ≈ 0.01 (1% daily vol) */
    return (rbpf_real_t)(0.01 * pcg32_gaussian(&g_rng));
}

static rbpf_real_t gen_trend_return(void) {
    /* σ ≈ 0.02 (2% daily vol) */
    return (rbpf_real_t)(0.02 * pcg32_gaussian(&g_rng));
}

static rbpf_real_t gen_crisis_return(void) {
    /* σ ≈ 0.05 (5% daily vol) */
    return (rbpf_real_t)(0.05 * pcg32_gaussian(&g_rng));
}

static rbpf_real_t gen_outlier_return(double sigma_mult) {
    /* Generate N-sigma outlier */
    double sign = (pcg32_double(&g_rng) < 0.5) ? -1.0 : 1.0;
    return (rbpf_real_t)(sign * sigma_mult * 0.01);  /* Base vol = 1% */
}

/*─────────────────────────────────────────────────────────────────────────────
 * HELPER: Create default MMPF
 *───────────────────────────────────────────────────────────────────────────*/

static MMPF_ROCKS *create_test_mmpf(void) {
    MMPF_Config cfg = mmpf_config_defaults();
    cfg.n_particles = 256;  /* Smaller for faster tests */
    return mmpf_create(&cfg);
}

/*═══════════════════════════════════════════════════════════════════════════
 * P0 TESTS: INVARIANTS
 *
 * These must ALWAYS hold, checked after every step.
 *═══════════════════════════════════════════════════════════════════════════*/

static void check_invariants(MMPF_ROCKS *mmpf, const char *context) {
    char msg[256];
    
    /* Model weights sum to 1 */
    rbpf_real_t weights[MMPF_N_MODELS];
    mmpf_get_weights(mmpf, weights);
    rbpf_real_t weight_sum = weights[0] + weights[1] + weights[2];
    snprintf(msg, sizeof(msg), "[%s] Model weights sum to 1", context);
    ASSERT_NEAR(weight_sum, 1.0, 1e-5, msg);
    
    /* All weights non-negative */
    for (int k = 0; k < MMPF_N_MODELS; k++) {
        snprintf(msg, sizeof(msg), "[%s] Weight[%d] >= 0", context, k);
        ASSERT_GE(weights[k], 0.0, msg);
    }
    
    /* Volatility positive */
    rbpf_real_t vol = mmpf_get_volatility(mmpf);
    snprintf(msg, sizeof(msg), "[%s] Volatility > 0", context);
    ASSERT_GT(vol, 0.0, msg);
    ASSERT_FINITE(vol, msg);
    
    /* Log volatility finite */
    rbpf_real_t log_vol = mmpf_get_log_volatility(mmpf);
    snprintf(msg, sizeof(msg), "[%s] Log volatility finite", context);
    ASSERT_FINITE(log_vol, msg);
    
    /* Volatility std non-negative */
    rbpf_real_t vol_std = mmpf_get_volatility_std(mmpf);
    snprintf(msg, sizeof(msg), "[%s] Vol std >= 0", context);
    ASSERT_GE(vol_std, 0.0, msg);
    ASSERT_FINITE(vol_std, msg);
    
    /* Outlier fraction in [0, 1] */
    rbpf_real_t outlier_frac = mmpf_get_outlier_fraction(mmpf);
    snprintf(msg, sizeof(msg), "[%s] Outlier fraction >= 0", context);
    ASSERT_GE(outlier_frac, 0.0, msg);
    snprintf(msg, sizeof(msg), "[%s] Outlier fraction <= 1", context);
    ASSERT_LE(outlier_frac, 1.0, msg);
    
    /* Dominant hypothesis valid */
    MMPF_Hypothesis dom = mmpf_get_dominant(mmpf);
    snprintf(msg, sizeof(msg), "[%s] Dominant in [0,2]", context);
    ASSERT_GE((int)dom, 0, msg);
    ASSERT_LE((int)dom, 2, msg);
    
    /* Dominant probability in (0, 1] */
    rbpf_real_t dom_prob = mmpf_get_dominant_probability(mmpf);
    snprintf(msg, sizeof(msg), "[%s] Dominant prob > 0", context);
    ASSERT_GT(dom_prob, 0.0, msg);
    snprintf(msg, sizeof(msg), "[%s] Dominant prob <= 1", context);
    ASSERT_LE(dom_prob, 1.0, msg);
    
    /* Stickiness in valid range */
    rbpf_real_t stickiness = mmpf_get_stickiness(mmpf);
    snprintf(msg, sizeof(msg), "[%s] Stickiness in [0.5, 1.0]", context);
    ASSERT_GE(stickiness, 0.5, msg);
    ASSERT_LE(stickiness, 1.0, msg);
}

static void test_invariants_after_create(void) {
    TEST_BEGIN("Invariants after create");
    
    MMPF_ROCKS *mmpf = create_test_mmpf();
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        check_invariants(mmpf, "after_create");
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

static void test_invariants_after_reset(void) {
    TEST_BEGIN("Invariants after reset");
    
    MMPF_ROCKS *mmpf = create_test_mmpf();
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        mmpf_reset(mmpf, RBPF_REAL(-3.0));  /* Initial log-vol */
        check_invariants(mmpf, "after_reset");
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

static void test_invariants_during_run(void) {
    TEST_BEGIN("Invariants during run (100 steps)");
    
    MMPF_ROCKS *mmpf = create_test_mmpf();
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        MMPF_Output out;
        for (int t = 0; t < 100; t++) {
            rbpf_real_t ret = gen_calm_return();
            mmpf_step(mmpf, ret, &out);
            
            char ctx[32];
            snprintf(ctx, sizeof(ctx), "step_%d", t);
            check_invariants(mmpf, ctx);
            
            if (g_current_test_failed) break;
        }
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

/*═══════════════════════════════════════════════════════════════════════════
 * P0 TESTS: ACCESSOR SIDE-EFFECTS
 *
 * Getters must be const-correct and not modify state.
 * This was a bug we just fixed!
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_accessor_no_side_effects(void) {
    TEST_BEGIN("Accessors have no side effects");
    
    MMPF_ROCKS *mmpf = create_test_mmpf();
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        MMPF_Output out;
        
        /* Run a few steps to get into steady state */
        for (int t = 0; t < 50; t++) {
            mmpf_step(mmpf, gen_calm_return(), &out);
        }
        
        /* Get values - first call */
        rbpf_real_t vol1 = mmpf_get_volatility(mmpf);
        rbpf_real_t log_vol1 = mmpf_get_log_volatility(mmpf);
        rbpf_real_t vol_std1 = mmpf_get_volatility_std(mmpf);
        rbpf_real_t outlier1 = mmpf_get_outlier_fraction(mmpf);
        MMPF_Hypothesis dom1 = mmpf_get_dominant(mmpf);
        rbpf_real_t dom_prob1 = mmpf_get_dominant_probability(mmpf);
        rbpf_real_t stick1 = mmpf_get_stickiness(mmpf);
        rbpf_real_t weights1[3];
        mmpf_get_weights(mmpf, weights1);
        
        /* Call all getters again multiple times */
        for (int i = 0; i < 10; i++) {
            (void)mmpf_get_volatility(mmpf);
            (void)mmpf_get_log_volatility(mmpf);
            (void)mmpf_get_volatility_std(mmpf);
            (void)mmpf_get_outlier_fraction(mmpf);
            (void)mmpf_get_dominant(mmpf);
            (void)mmpf_get_dominant_probability(mmpf);
            (void)mmpf_get_stickiness(mmpf);
            rbpf_real_t tmp[3];
            mmpf_get_weights(mmpf, tmp);
        }
        
        /* Get values - after repeated calls */
        rbpf_real_t vol2 = mmpf_get_volatility(mmpf);
        rbpf_real_t log_vol2 = mmpf_get_log_volatility(mmpf);
        rbpf_real_t vol_std2 = mmpf_get_volatility_std(mmpf);
        rbpf_real_t outlier2 = mmpf_get_outlier_fraction(mmpf);
        MMPF_Hypothesis dom2 = mmpf_get_dominant(mmpf);
        rbpf_real_t dom_prob2 = mmpf_get_dominant_probability(mmpf);
        rbpf_real_t stick2 = mmpf_get_stickiness(mmpf);
        rbpf_real_t weights2[3];
        mmpf_get_weights(mmpf, weights2);
        
        /* All values must be identical */
        ASSERT_EQ_INT((int)(vol1 * 1e6), (int)(vol2 * 1e6), "Volatility changed");
        ASSERT_EQ_INT((int)(log_vol1 * 1e6), (int)(log_vol2 * 1e6), "Log volatility changed");
        ASSERT_EQ_INT((int)(vol_std1 * 1e6), (int)(vol_std2 * 1e6), "Vol std changed");
        ASSERT_EQ_INT((int)(outlier1 * 1e6), (int)(outlier2 * 1e6), "Outlier frac changed");
        ASSERT_EQ_INT((int)dom1, (int)dom2, "Dominant changed");
        ASSERT_EQ_INT((int)(dom_prob1 * 1e6), (int)(dom_prob2 * 1e6), "Dominant prob changed");
        ASSERT_EQ_INT((int)(stick1 * 1e6), (int)(stick2 * 1e6), "Stickiness changed");
        
        for (int k = 0; k < 3; k++) {
            char msg[64];
            snprintf(msg, sizeof(msg), "Weight[%d] changed", k);
            ASSERT_EQ_INT((int)(weights1[k] * 1e6), (int)(weights2[k] * 1e6), msg);
        }
        
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

static void test_accessor_after_step_consistency(void) {
    TEST_BEGIN("Accessor values match step output");
    
    MMPF_ROCKS *mmpf = create_test_mmpf();
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        MMPF_Output out;
        
        /* Run some steps and verify output matches accessors */
        for (int t = 0; t < 20; t++) {
            mmpf_step(mmpf, gen_trend_return(), &out);
            
            /* Compare output struct to accessor values */
            ASSERT_NEAR(out.volatility, mmpf_get_volatility(mmpf), 1e-6,
                       "Output.volatility != get_volatility()");
            ASSERT_NEAR(out.log_volatility, mmpf_get_log_volatility(mmpf), 1e-6,
                       "Output.log_volatility != get_log_volatility()");
            ASSERT_EQ_INT((int)out.dominant, (int)mmpf_get_dominant(mmpf),
                         "Output.dominant != get_dominant()");
            
            if (g_current_test_failed) break;
        }
        
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

/*═══════════════════════════════════════════════════════════════════════════
 * P0 TESTS: PARAMETER SYNC
 *
 * Verify that Storvik's learned parameters actually reach RBPF.
 * This was a critical bug we just fixed!
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_parameter_sync_after_step(void) {
    TEST_BEGIN("Parameters sync from Storvik to RBPF");
    
    MMPF_ROCKS *mmpf = create_test_mmpf();
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        MMPF_Output out;
        
        /* Run enough steps for Storvik to update parameters */
        for (int t = 0; t < 200; t++) {
            mmpf_step(mmpf, gen_calm_return(), &out);
        }
        
        /* Check that RBPF is using learned params */
        for (int k = 0; k < MMPF_N_MODELS; k++) {
            RBPF_KSC *rbpf = mmpf->rbpf[k];
            
            char msg[64];
            snprintf(msg, sizeof(msg), "Model[%d] use_learned_params == 1", k);
            ASSERT_EQ_INT(rbpf->use_learned_params, 1, msg);
            
            /* Verify particle_mu_vol is populated (not all zeros) */
            int n = rbpf->n_particles;
            int n_regimes = rbpf->n_regimes;
            double sum = 0;
            for (int i = 0; i < n * n_regimes; i++) {
                sum += fabs(rbpf->particle_mu_vol[i]);
            }
            snprintf(msg, sizeof(msg), "Model[%d] particle_mu_vol populated", k);
            ASSERT_GT(sum, 0.0, msg);
        }
        
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

static void test_parameter_sync_type_conversion(void) {
    TEST_BEGIN("Double to float conversion correct");
    
    MMPF_ROCKS *mmpf = create_test_mmpf();
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        MMPF_Output out;
        
        /* Run steps */
        for (int t = 0; t < 100; t++) {
            mmpf_step(mmpf, gen_calm_return(), &out);
        }
        
        /* Check values are in reasonable range (not garbage from bad memcpy) */
        for (int k = 0; k < MMPF_N_MODELS; k++) {
            RBPF_KSC *rbpf = mmpf->rbpf[k];
            int n = rbpf->n_particles;
            int n_regimes = rbpf->n_regimes;
            
            for (int i = 0; i < n * n_regimes; i++) {
                rbpf_real_t mu = rbpf->particle_mu_vol[i];
                rbpf_real_t sigma = rbpf->particle_sigma_vol[i];
                
                char msg[128];
                
                /* mu_vol should be in [-10, 2] range typically */
                snprintf(msg, sizeof(msg), "Model[%d] mu_vol[%d] in range", k, i);
                ASSERT_GT(mu, -15.0, msg);
                ASSERT_LE(mu, 5.0, msg);
                ASSERT_FINITE(mu, msg);
                
                /* sigma_vol should be positive and reasonable */
                snprintf(msg, sizeof(msg), "Model[%d] sigma_vol[%d] in range", k, i);
                ASSERT_GT(sigma, 0.0, msg);
                ASSERT_LE(sigma, 5.0, msg);
                ASSERT_FINITE(sigma, msg);
                
                if (g_current_test_failed) break;
            }
            if (g_current_test_failed) break;
        }
        
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

/*═══════════════════════════════════════════════════════════════════════════
 * P0 TESTS: IMM MIXING MATH
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_imm_mixing_weights_valid(void) {
    TEST_BEGIN("IMM mixing preserves particle count");
    
    MMPF_ROCKS *mmpf = create_test_mmpf();
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        MMPF_Output out;
        
        /* Run steps and check mixing counts sum correctly */
        for (int t = 0; t < 50; t++) {
            mmpf_step(mmpf, gen_calm_return(), &out);
            
            /* After each step, all models should have correct particle count */
            for (int k = 0; k < MMPF_N_MODELS; k++) {
                int n = mmpf->rbpf[k]->n_particles;
                char msg[64];
                snprintf(msg, sizeof(msg), "Model[%d] particle count", k);
                ASSERT_EQ_INT(n, mmpf->config.n_particles, msg);
            }
            
            if (g_current_test_failed) break;
        }
        
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

/*═══════════════════════════════════════════════════════════════════════════
 * P1 TESTS: HYPOTHESIS RECOVERY
 *
 * Core IMM value proposition: a "dead" hypothesis can recover.
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_hypothesis_recovery(void) {
    TEST_BEGIN("Dead hypothesis can recover via IMM");
    
    MMPF_ROCKS *mmpf = create_test_mmpf();
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        MMPF_Output out;
        
        /* Run calm data to suppress Crisis hypothesis */
        for (int t = 0; t < 200; t++) {
            mmpf_step(mmpf, gen_calm_return(), &out);
        }
        
        rbpf_real_t weights_before[3];
        mmpf_get_weights(mmpf, weights_before);
        
        /* Crisis should be suppressed */
        ASSERT_LE(weights_before[MMPF_CRISIS], 0.1, 
                 "Crisis not suppressed after calm data");
        
        /* But not completely dead (IMM keeps it alive) */
        ASSERT_GT(weights_before[MMPF_CRISIS], 0.001, 
                 "Crisis completely dead - IMM not working?");
        
        /* Now inject crisis data */
        for (int t = 0; t < 100; t++) {
            mmpf_step(mmpf, gen_crisis_return(), &out);
        }
        
        rbpf_real_t weights_after[3];
        mmpf_get_weights(mmpf, weights_after);
        
        /* Crisis should recover */
        ASSERT_GT(weights_after[MMPF_CRISIS], 0.3,
                 "Crisis failed to recover after crisis data");
        
        printf("\n    Crisis weight: %.4f → %.4f (recovered)\n", 
               weights_before[MMPF_CRISIS], weights_after[MMPF_CRISIS]);
        
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

static void test_all_hypotheses_stay_alive(void) {
    TEST_BEGIN("All hypotheses stay alive during extended calm");
    
    MMPF_ROCKS *mmpf = create_test_mmpf();
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        MMPF_Output out;
        
        /* Run 1000 steps of calm data */
        for (int t = 0; t < 1000; t++) {
            mmpf_step(mmpf, gen_calm_return(), &out);
        }
        
        rbpf_real_t weights[3];
        mmpf_get_weights(mmpf, weights);
        
        /* All hypotheses should still have non-negligible weight */
        for (int k = 0; k < 3; k++) {
            char msg[64];
            snprintf(msg, sizeof(msg), "Hypothesis[%d] not dead after 1000 calm steps", k);
            ASSERT_GT(weights[k], 1e-6, msg);
        }
        
        /* And ESS should be healthy for all models */
        for (int k = 0; k < 3; k++) {
            rbpf_real_t ess = mmpf->model_output[k].ess;
            char msg[64];
            snprintf(msg, sizeof(msg), "Model[%d] ESS healthy", k);
            ASSERT_GT(ess, 10.0, msg);  /* Not collapsed */
        }
        
        printf("\n    After 1000 calm: w=[%.4f, %.4f, %.4f]\n",
               weights[0], weights[1], weights[2]);
        
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

/*═══════════════════════════════════════════════════════════════════════════
 * P1 TESTS: SWITCHING SPEED
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_switching_speed_calm_to_crisis(void) {
    TEST_BEGIN("Switching speed: Calm → Crisis");
    
    MMPF_ROCKS *mmpf = create_test_mmpf();
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        MMPF_Output out;
        
        /* Burn-in: establish calm */
        for (int t = 0; t < 300; t++) {
            mmpf_step(mmpf, gen_calm_return(), &out);
        }
        
        MMPF_Hypothesis dom_before = mmpf_get_dominant(mmpf);
        ASSERT_EQ_INT((int)dom_before, (int)MMPF_CALM, "Should be Calm before switch");
        
        /* Switch to crisis and count detection lag */
        int ticks_to_detect = 0;
        int detected = 0;
        for (int t = 0; t < 100; t++) {
            mmpf_step(mmpf, gen_crisis_return(), &out);
            ticks_to_detect++;
            
            if (mmpf_get_dominant(mmpf) == MMPF_CRISIS) {
                detected = 1;
                break;
            }
        }
        
        ASSERT_TRUE(detected, "Failed to detect crisis within 100 ticks");
        ASSERT_LE(ticks_to_detect, 30, "Detection too slow (> 30 ticks)");
        
        printf("\n    Detection lag: %d ticks\n", ticks_to_detect);
        
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

static void test_switching_speed_crisis_to_calm(void) {
    TEST_BEGIN("Switching speed: Crisis → Calm");
    
    MMPF_ROCKS *mmpf = create_test_mmpf();
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        MMPF_Output out;
        
        /* Burn-in: establish crisis */
        for (int t = 0; t < 300; t++) {
            mmpf_step(mmpf, gen_crisis_return(), &out);
        }
        
        MMPF_Hypothesis dom_before = mmpf_get_dominant(mmpf);
        ASSERT_EQ_INT((int)dom_before, (int)MMPF_CRISIS, "Should be Crisis before switch");
        
        /* Switch to calm and count detection lag */
        int ticks_to_detect = 0;
        int detected = 0;
        for (int t = 0; t < 200; t++) {
            mmpf_step(mmpf, gen_calm_return(), &out);
            ticks_to_detect++;
            
            if (mmpf_get_dominant(mmpf) == MMPF_CALM) {
                detected = 1;
                break;
            }
        }
        
        ASSERT_TRUE(detected, "Failed to detect calm within 200 ticks");
        /* Recovery can be slower (crisis is sticky by design) */
        ASSERT_LE(ticks_to_detect, 100, "Recovery too slow (> 100 ticks)");
        
        printf("\n    Recovery lag: %d ticks\n", ticks_to_detect);
        
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

/*═══════════════════════════════════════════════════════════════════════════
 * P1 TESTS: OCSN INTEGRATION
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_ocsn_detects_outliers(void) {
    TEST_BEGIN("OCSN detects injected outliers");
    
    MMPF_Config cfg = mmpf_config_defaults();
    cfg.n_particles = 256;
    cfg.robust_ocsn.enabled = 1;
    
    MMPF_ROCKS *mmpf = mmpf_create(&cfg);
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        MMPF_Output out;
        
        /* Burn-in */
        for (int t = 0; t < 100; t++) {
            mmpf_step(mmpf, gen_calm_return(), &out);
        }
        
        /* Get baseline outlier fraction on normal data */
        double sum_normal = 0;
        for (int t = 0; t < 50; t++) {
            mmpf_step(mmpf, gen_calm_return(), &out);
            sum_normal += mmpf_get_outlier_fraction(mmpf);
        }
        double avg_normal = sum_normal / 50;
        
        /* Inject outlier */
        mmpf_step(mmpf, gen_outlier_return(10.0), &out);  /* 10σ */
        double outlier_frac = mmpf_get_outlier_fraction(mmpf);
        
        printf("\n    Avg normal: %.4f, On 10σ outlier: %.4f\n", 
               avg_normal, outlier_frac);
        
        /* Outlier fraction should be elevated */
        ASSERT_GT(outlier_frac, avg_normal + 0.1,
                 "Outlier fraction not elevated on 10σ move");
        ASSERT_GT(outlier_frac, 0.3,
                 "Outlier fraction too low on 10σ move");
        
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

static void test_ocsn_adaptive_stickiness(void) {
    TEST_BEGIN("OCSN drives adaptive stickiness");
    
    MMPF_Config cfg = mmpf_config_defaults();
    cfg.n_particles = 256;
    cfg.robust_ocsn.enabled = 1;
    cfg.enable_adaptive_stickiness = 1;
    
    MMPF_ROCKS *mmpf = mmpf_create(&cfg);
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        MMPF_Output out;
        
        /* Burn-in with calm data */
        for (int t = 0; t < 100; t++) {
            mmpf_step(mmpf, gen_calm_return(), &out);
        }
        
        rbpf_real_t stick_calm = mmpf_get_stickiness(mmpf);
        
        /* Inject several outliers to increase novelty */
        for (int t = 0; t < 10; t++) {
            mmpf_step(mmpf, gen_outlier_return(8.0), &out);
        }
        
        rbpf_real_t stick_after = mmpf_get_stickiness(mmpf);
        
        printf("\n    Stickiness: calm=%.4f, after_outliers=%.4f\n",
               stick_calm, stick_after);
        
        /* Stickiness should decrease when outlier fraction is high */
        ASSERT_LE(stick_after, stick_calm,
                 "Stickiness should decrease with high outlier fraction");
        
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

/*═══════════════════════════════════════════════════════════════════════════
 * P2 TESTS: EXTREME INPUTS
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_extreme_zero_return(void) {
    TEST_BEGIN("Handles zero return");
    
    MMPF_ROCKS *mmpf = create_test_mmpf();
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        MMPF_Output out;
        
        /* Burn-in */
        for (int t = 0; t < 50; t++) {
            mmpf_step(mmpf, gen_calm_return(), &out);
        }
        
        /* Zero return */
        mmpf_step(mmpf, RBPF_REAL(0.0), &out);
        check_invariants(mmpf, "zero_return");
        
        /* Multiple zeros */
        for (int t = 0; t < 10; t++) {
            mmpf_step(mmpf, RBPF_REAL(0.0), &out);
            check_invariants(mmpf, "multi_zero");
        }
        
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

static void test_extreme_tiny_return(void) {
    TEST_BEGIN("Handles tiny return (1e-15)");
    
    MMPF_ROCKS *mmpf = create_test_mmpf();
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        MMPF_Output out;
        
        /* Burn-in */
        for (int t = 0; t < 50; t++) {
            mmpf_step(mmpf, gen_calm_return(), &out);
        }
        
        /* Tiny returns */
        for (int t = 0; t < 10; t++) {
            mmpf_step(mmpf, RBPF_REAL(1e-15), &out);
            check_invariants(mmpf, "tiny_return");
        }
        
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

static void test_extreme_huge_return(void) {
    TEST_BEGIN("Handles huge return (50%)");
    
    MMPF_ROCKS *mmpf = create_test_mmpf();
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        MMPF_Output out;
        
        /* Burn-in */
        for (int t = 0; t < 50; t++) {
            mmpf_step(mmpf, gen_calm_return(), &out);
        }
        
        /* Huge returns */
        mmpf_step(mmpf, RBPF_REAL(0.5), &out);  /* +50% */
        check_invariants(mmpf, "huge_pos");
        
        mmpf_step(mmpf, RBPF_REAL(-0.5), &out);  /* -50% */
        check_invariants(mmpf, "huge_neg");
        
        /* Resume normal operation */
        for (int t = 0; t < 20; t++) {
            mmpf_step(mmpf, gen_calm_return(), &out);
            check_invariants(mmpf, "post_huge");
        }
        
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

static void test_extreme_consecutive_outliers(void) {
    TEST_BEGIN("Handles 20 consecutive outliers");
    
    MMPF_ROCKS *mmpf = create_test_mmpf();
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        MMPF_Output out;
        
        /* Burn-in */
        for (int t = 0; t < 50; t++) {
            mmpf_step(mmpf, gen_calm_return(), &out);
        }
        
        /* 20 consecutive 10σ outliers */
        for (int t = 0; t < 20; t++) {
            double sign = (t % 2) ? 1.0 : -1.0;
            mmpf_step(mmpf, (rbpf_real_t)(sign * 0.10), &out);
            
            char ctx[32];
            snprintf(ctx, sizeof(ctx), "outlier_%d", t);
            check_invariants(mmpf, ctx);
            
            if (g_current_test_failed) break;
        }
        
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

/*═══════════════════════════════════════════════════════════════════════════
 * P2 TESTS: NUMERICAL STABILITY
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_numerical_no_nan(void) {
    TEST_BEGIN("No NaN after 10000 steps");
    
    MMPF_ROCKS *mmpf = create_test_mmpf();
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        MMPF_Output out;
        int nan_found = 0;
        
        for (int t = 0; t < 10000 && !nan_found; t++) {
            /* Mix of different return magnitudes */
            rbpf_real_t ret;
            if (t % 100 == 50) {
                ret = gen_outlier_return(8.0);  /* Occasional outlier */
            } else if (t % 500 < 100) {
                ret = gen_crisis_return();  /* Crisis periods */
            } else {
                ret = gen_calm_return();  /* Mostly calm */
            }
            
            mmpf_step(mmpf, ret, &out);
            
            /* Check for NaN */
            if (isnan(out.volatility) || isnan(out.log_volatility)) {
                nan_found = 1;
                printf("\n    NaN found at tick %d\n", t);
            }
            
            for (int k = 0; k < 3 && !nan_found; k++) {
                rbpf_real_t w;
                rbpf_real_t weights[3];
                mmpf_get_weights(mmpf, weights);
                w = weights[k];
                if (isnan(w)) {
                    nan_found = 1;
                    printf("\n    NaN in weight[%d] at tick %d\n", k, t);
                }
            }
        }
        
        ASSERT_TRUE(!nan_found, "NaN detected during run");
        
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

static void test_numerical_log_weight_bounded(void) {
    TEST_BEGIN("Log weights stay bounded after 50000 steps");
    
    MMPF_ROCKS *mmpf = create_test_mmpf();
    ASSERT_TRUE(mmpf != NULL, "mmpf_create returned NULL");
    
    if (mmpf) {
        MMPF_Output out;
        
        for (int t = 0; t < 50000; t++) {
            rbpf_real_t ret = gen_calm_return();
            mmpf_step(mmpf, ret, &out);
        }
        
        /* Log weights should be normalized (not exploding) */
        for (int k = 0; k < 3; k++) {
            rbpf_real_t lw = mmpf->log_weights[k];
            char msg[64];
            
            snprintf(msg, sizeof(msg), "log_weight[%d] finite", k);
            ASSERT_FINITE(lw, msg);
            
            snprintf(msg, sizeof(msg), "log_weight[%d] bounded below", k);
            ASSERT_GT(lw, -100.0, msg);
            
            snprintf(msg, sizeof(msg), "log_weight[%d] bounded above", k);
            ASSERT_LE(lw, 100.0, msg);
        }
        
        mmpf_destroy(mmpf);
    }
    
    TEST_END();
}

/*═══════════════════════════════════════════════════════════════════════════
 * P3 TESTS: DETERMINISM
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_determinism(void) {
    TEST_BEGIN("Same seed produces same results");
    
    MMPF_Config cfg = mmpf_config_defaults();
    cfg.n_particles = 128;
    cfg.rng_seed = 12345;
    
    /* First run */
    MMPF_ROCKS *mmpf1 = mmpf_create(&cfg);
    ASSERT_TRUE(mmpf1 != NULL, "mmpf_create returned NULL");
    
    MMPF_Output out1[100];
    rng_init(42);  /* Fixed seed for returns */
    for (int t = 0; t < 100; t++) {
        mmpf_step(mmpf1, gen_calm_return(), &out1[t]);
    }
    
    /* Second run with same config */
    MMPF_ROCKS *mmpf2 = mmpf_create(&cfg);
    ASSERT_TRUE(mmpf2 != NULL, "mmpf_create returned NULL");
    
    MMPF_Output out2[100];
    rng_init(42);  /* Same seed for returns */
    for (int t = 0; t < 100; t++) {
        mmpf_step(mmpf2, gen_calm_return(), &out2[t]);
    }
    
    /* Compare outputs */
    int mismatch = 0;
    for (int t = 0; t < 100 && !mismatch; t++) {
        if (fabs(out1[t].volatility - out2[t].volatility) > 1e-6) {
            mismatch = 1;
            printf("\n    Mismatch at tick %d: vol1=%.6f, vol2=%.6f\n",
                   t, out1[t].volatility, out2[t].volatility);
        }
        if (out1[t].dominant != out2[t].dominant) {
            mismatch = 1;
            printf("\n    Mismatch at tick %d: dom1=%d, dom2=%d\n",
                   t, out1[t].dominant, out2[t].dominant);
        }
    }
    
    ASSERT_TRUE(!mismatch, "Results differ between runs with same seed");
    
    if (mmpf1) mmpf_destroy(mmpf1);
    if (mmpf2) mmpf_destroy(mmpf2);
    
    TEST_END();
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST RUNNER
 *═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char **argv) {
    int seed = 42;
    if (argc > 1) {
        seed = atoi(argv[1]);
    }
    
    rng_init(seed);
    
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  MMPF Correctness Test Suite\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Seed: %d\n\n", seed);
    
    /*─────────────────────────────────────────────────────────────────────*/
    printf("P0: INVARIANT TESTS\n");
    printf("───────────────────────────────────────────────────────────────\n");
    
    test_invariants_after_create();
    test_invariants_after_reset();
    test_invariants_during_run();
    
    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nP0: ACCESSOR SIDE-EFFECT TESTS\n");
    printf("───────────────────────────────────────────────────────────────\n");
    
    test_accessor_no_side_effects();
    test_accessor_after_step_consistency();
    
    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nP0: PARAMETER SYNC TESTS\n");
    printf("───────────────────────────────────────────────────────────────\n");
    
    test_parameter_sync_after_step();
    test_parameter_sync_type_conversion();
    
    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nP0: IMM MIXING TESTS\n");
    printf("───────────────────────────────────────────────────────────────\n");
    
    test_imm_mixing_weights_valid();
    
    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nP1: HYPOTHESIS RECOVERY TESTS\n");
    printf("───────────────────────────────────────────────────────────────\n");
    
    test_hypothesis_recovery();
    test_all_hypotheses_stay_alive();
    
    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nP1: SWITCHING SPEED TESTS\n");
    printf("───────────────────────────────────────────────────────────────\n");
    
    test_switching_speed_calm_to_crisis();
    test_switching_speed_crisis_to_calm();
    
    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nP1: OCSN INTEGRATION TESTS\n");
    printf("───────────────────────────────────────────────────────────────\n");
    
    test_ocsn_detects_outliers();
    test_ocsn_adaptive_stickiness();
    
    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nP2: EXTREME INPUT TESTS\n");
    printf("───────────────────────────────────────────────────────────────\n");
    
    test_extreme_zero_return();
    test_extreme_tiny_return();
    test_extreme_huge_return();
    test_extreme_consecutive_outliers();
    
    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nP2: NUMERICAL STABILITY TESTS\n");
    printf("───────────────────────────────────────────────────────────────\n");
    
    test_numerical_no_nan();
    test_numerical_log_weight_bounded();
    
    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nP3: DETERMINISM TESTS\n");
    printf("───────────────────────────────────────────────────────────────\n");
    
    test_determinism();
    
    /*─────────────────────────────────────────────────────────────────────*/
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Tests run:    %d\n", g_tests_run);
    printf("  Passed:       %d\n", g_tests_passed);
    printf("  Failed:       %d\n", g_tests_failed);
    printf("═══════════════════════════════════════════════════════════════\n");
    
    if (g_tests_failed > 0) {
        printf("\n  *** %d TEST(S) FAILED ***\n\n", g_tests_failed);
        return 1;
    } else {
        printf("\n  All tests passed!\n\n");
        return 0;
    }
}
