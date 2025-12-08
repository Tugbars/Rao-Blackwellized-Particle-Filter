/**
 * @file test_rbpf_integration.c
 * @brief Test/Benchmark: RBPF-KSC + Sleeping Storvik Integration (Option B)
 *
 * Option B Architecture:
 *   - use_learned_params: Controls whether predict() reads particle_mu_vol[]
 *   - liu_west.enabled:   Controls whether Liu-West update/resample runs
 *
 *   STORVIK mode:  use_learned_params=1, liu_west.enabled=0
 *   LIU_WEST mode: use_learned_params=1, liu_west.enabled=1
 *   HYBRID mode:   use_learned_params=1, liu_west.enabled=1
 *
 * Tests:
 *   1. Initialization and Option B flag configuration
 *   2. Regime params sync (RBPF ↔ Storvik)
 *   3. Basic update flow with stats verification
 *   4. Structural break trigger (SSA → immediate sampling)
 *   5. HFT mode interval adjustment
 *   6. Per-particle params active (Option B regression test)
 *
 * Benchmarks:
 *   1. Latency comparison (Liu-West vs Storvik vs Hybrid)
 *   2. Sleeping behavior (R0 vs R3 sampling frequency)
 *
 * Build (requires MKL):
 *   icc -O3 -xHost -qopenmp test_rbpf_integration.c rbpf_ksc_param_integration.c \
 *       rbpf_ksc.c rbpf_param_learn.c -o test_rbpf_int -lmkl_rt -lm
 *
 * Or with GCC + MKL:
 *   gcc -O3 -march=native -fopenmp test_rbpf_integration.c rbpf_ksc_param_integration.c \
 *       rbpf_ksc.c rbpf_param_learn.c -o test_rbpf_int \
 *       -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_rt -lm -lpthread
 */

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "rbpf_ksc_param_integration.h"

/*═══════════════════════════════════════════════════════════════════════════
 * PLATFORM-SPECIFIC TIMING
 *═══════════════════════════════════════════════════════════════════════════*/

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

typedef struct
{
    LARGE_INTEGER start;
    LARGE_INTEGER end;
} Timer;

static LARGE_INTEGER timer_freq = {0};

static void timer_start(Timer *t)
{
    if (timer_freq.QuadPart == 0)
    {
        QueryPerformanceFrequency(&timer_freq);
    }
    QueryPerformanceCounter(&t->start);
}

static double timer_stop_us(Timer *t)
{
    QueryPerformanceCounter(&t->end);
    return (double)(t->end.QuadPart - t->start.QuadPart) * 1e6 / (double)timer_freq.QuadPart;
}
#else
typedef struct
{
    struct timespec start;
    struct timespec end;
} Timer;

static void timer_start(Timer *t)
{
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}

static double timer_stop_us(Timer *t)
{
    clock_gettime(CLOCK_MONOTONIC, &t->end);
    double sec = (t->end.tv_sec - t->start.tv_sec);
    double nsec = (t->end.tv_nsec - t->start.tv_nsec);
    return (sec * 1e6) + (nsec / 1000.0);
}
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * TEST HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

#define TEST_PASS "\033[32m✓\033[0m"
#define TEST_FAIL "\033[31m✗\033[0m"

static int g_passed = 0;
static int g_failed = 0;

#define ASSERT(cond, msg)                                 \
    do                                                    \
    {                                                     \
        if (cond)                                         \
        {                                                 \
            printf("  %s %s\n", TEST_PASS, msg);          \
            g_passed++;                                   \
        }                                                 \
        else                                              \
        {                                                 \
            printf("  %s %s (FAILED)\n", TEST_FAIL, msg); \
            g_failed++;                                   \
        }                                                 \
    } while (0)

#define ASSERT_NEAR(a, b, tol, msg)                                                                         \
    do                                                                                                      \
    {                                                                                                       \
        double _diff = fabs((double)(a) - (double)(b));                                                     \
        if (_diff <= (tol))                                                                                 \
        {                                                                                                   \
            printf("  %s %s (%.4f ≈ %.4f)\n", TEST_PASS, msg, (double)(a), (double)(b));                    \
            g_passed++;                                                                                     \
        }                                                                                                   \
        else                                                                                                \
        {                                                                                                   \
            printf("  %s %s (%.4f != %.4f, diff=%.2e)\n", TEST_FAIL, msg, (double)(a), (double)(b), _diff); \
            g_failed++;                                                                                     \
        }                                                                                                   \
    } while (0)

/* Simple RNG for test data */
static uint64_t test_rng[2] = {0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL};

static double test_randn(void)
{
    uint64_t s0 = test_rng[0];
    uint64_t s1 = test_rng[1];
    uint64_t result = s0 + s1;
    s1 ^= s0;
    test_rng[0] = ((s0 << 24) | (s0 >> 40)) ^ s1 ^ (s1 << 16);
    test_rng[1] = (s1 << 37) | (s1 >> 27);

    double u1 = (result >> 11) * (1.0 / 9007199254740992.0);

    s0 = test_rng[0];
    s1 = test_rng[1];
    result = s0 + s1;
    s1 ^= s0;
    test_rng[0] = ((s0 << 24) | (s0 >> 40)) ^ s1 ^ (s1 << 16);
    test_rng[1] = (s1 << 37) | (s1 >> 27);

    double u2 = (result >> 11) * (1.0 / 9007199254740992.0);
    if (u1 < 1e-15)
        u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 1: INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_initialization(void)
{
    printf("\n═══ Test 1: Initialization ═══\n");

    /* Test each mode */
    RBPF_Extended *ext_lw = rbpf_ext_create(200, 4, RBPF_PARAM_LIU_WEST);
    ASSERT(ext_lw != NULL, "Create with Liu-West mode");
    ASSERT(ext_lw->rbpf != NULL, "RBPF core created");
    ASSERT(ext_lw->storvik_initialized == 0, "Storvik NOT initialized (Liu-West mode)");
    ASSERT(ext_lw->rbpf->liu_west.enabled == 1, "Liu-West enabled");
    ASSERT(ext_lw->rbpf->use_learned_params == 1, "use_learned_params == 1 (via Liu-West)");
    rbpf_ext_destroy(ext_lw);

    RBPF_Extended *ext_sv = rbpf_ext_create(200, 4, RBPF_PARAM_STORVIK);
    ASSERT(ext_sv != NULL, "Create with Storvik mode");
    ASSERT(ext_sv->storvik_initialized == 1, "Storvik initialized");
    ASSERT(ext_sv->particle_info != NULL, "Particle info buffer allocated");
    ASSERT(ext_sv->ell_lag_buffer != NULL, "Lag buffer allocated");
    /* OPTION B: Storvik mode uses per-particle arrays but NOT Liu-West logic */
    ASSERT(ext_sv->rbpf->use_learned_params == 1, "use_learned_params == 1 (Storvik)");
    ASSERT(ext_sv->rbpf->liu_west.enabled == 0, "liu_west.enabled == 0 (no wasted work)");
    rbpf_ext_destroy(ext_sv);

    RBPF_Extended *ext_hyb = rbpf_ext_create(200, 4, RBPF_PARAM_HYBRID);
    ASSERT(ext_hyb != NULL, "Create with Hybrid mode");
    ASSERT(ext_hyb->storvik_initialized == 1, "Storvik initialized in hybrid");
    ASSERT(ext_hyb->rbpf->liu_west.enabled == 1, "Liu-West enabled in hybrid");
    ASSERT(ext_hyb->rbpf->use_learned_params == 1, "use_learned_params == 1 (Hybrid)");
    rbpf_ext_destroy(ext_hyb);

    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 2: REGIME PARAMS SYNC
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_regime_params_sync(void)
{
    printf("═══ Test 2: Regime Params Sync ═══\n");

    RBPF_Extended *ext = rbpf_ext_create(100, 4, RBPF_PARAM_STORVIK);

    /* Set regime params through extended API */
    rbpf_ext_set_regime_params(ext, 0, 0.05f, -4.6f, 0.10f);
    rbpf_ext_set_regime_params(ext, 1, 0.08f, -3.5f, 0.15f);
    rbpf_ext_set_regime_params(ext, 2, 0.10f, -2.5f, 0.20f);
    rbpf_ext_set_regime_params(ext, 3, 0.15f, -1.5f, 0.30f);

    /* Check RBPF received params */
    ASSERT_NEAR(ext->rbpf->params[0].mu_vol, -4.6f, 0.01f, "RBPF R0 mu_vol set");
    ASSERT_NEAR(ext->rbpf->params[2].theta, 0.10f, 0.01f, "RBPF R2 theta set");

    /* Check Storvik received params */
    ASSERT_NEAR(ext->storvik.priors[0].m, -4.6f, 0.01f, "Storvik R0 m (mu prior) set");
    ASSERT_NEAR(ext->storvik.priors[2].phi, 0.10f, 0.01f, "Storvik R2 phi set");

    rbpf_ext_destroy(ext);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 3: BASIC UPDATE FLOW
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_basic_update(void)
{
    printf("═══ Test 3: Basic Update Flow ═══\n");

    RBPF_Extended *ext = rbpf_ext_create(100, 4, RBPF_PARAM_STORVIK);

    /* Set up regime params */
    rbpf_ext_set_regime_params(ext, 0, 0.05f, -4.6f, 0.10f);
    rbpf_ext_set_regime_params(ext, 1, 0.08f, -3.5f, 0.15f);
    rbpf_ext_set_regime_params(ext, 2, 0.10f, -2.5f, 0.20f);
    rbpf_ext_set_regime_params(ext, 3, 0.15f, -1.5f, 0.30f);

    /* Build transition LUT (diagonal dominant) */
    rbpf_real_t trans[16] = {
        0.95f, 0.04f, 0.01f, 0.00f,
        0.05f, 0.90f, 0.04f, 0.01f,
        0.01f, 0.05f, 0.90f, 0.04f,
        0.00f, 0.01f, 0.04f, 0.95f};
    rbpf_ext_build_transition_lut(ext, trans);

    /* Initialize */
    rbpf_ext_init(ext, -4.6f, 0.1f);

    uint64_t stats_before = ext->storvik.total_stat_updates;

    /* Run a few updates */
    RBPF_KSC_Output output;
    for (int t = 0; t < 100; t++)
    {
        rbpf_real_t ret = 0.01f * (rbpf_real_t)test_randn(); /* Simulated return */
        rbpf_ext_step(ext, ret, &output);
    }

    uint64_t stats_after = ext->storvik.total_stat_updates;

    ASSERT(stats_after > stats_before, "Storvik stats updated during loop");
    ASSERT(stats_after >= 100 * 100, "At least 100 updates × 100 particles");
    ASSERT(output.vol_mean > 0, "Vol mean is positive");
    ASSERT(output.ess > 0, "ESS is positive");

    /* Check learned params available */
    rbpf_real_t mu_vol, sigma_vol;
    rbpf_ext_get_learned_params(ext, 0, &mu_vol, &sigma_vol);
    ASSERT(mu_vol != 0, "Learned mu_vol available");
    ASSERT(sigma_vol > 0, "Learned sigma_vol positive");

    rbpf_ext_destroy(ext);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 4: STRUCTURAL BREAK TRIGGER
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_structural_break(void)
{
    printf("═══ Test 4: Structural Break Trigger ═══\n");

    RBPF_Extended *ext = rbpf_ext_create(100, 4, RBPF_PARAM_STORVIK);

    rbpf_ext_set_regime_params(ext, 0, 0.05f, -4.6f, 0.10f);
    rbpf_ext_set_regime_params(ext, 1, 0.08f, -3.5f, 0.15f);

    rbpf_real_t trans[16] = {
        0.95f, 0.05f, 0.00f, 0.00f,
        0.05f, 0.95f, 0.00f, 0.00f,
        0.00f, 0.00f, 1.00f, 0.00f,
        0.00f, 0.00f, 0.00f, 1.00f};
    rbpf_ext_build_transition_lut(ext, trans);
    rbpf_ext_init(ext, -4.6f, 0.1f);

    /* Run normal updates */
    RBPF_KSC_Output output;
    for (int t = 0; t < 50; t++)
    {
        rbpf_real_t ret = 0.01f * (rbpf_real_t)test_randn();
        rbpf_ext_step(ext, ret, &output);
    }

    uint64_t samples_before = ext->storvik.total_samples_drawn;
    uint64_t break_triggers_before = ext->storvik.samples_triggered_break;

    /* Signal structural break */
    rbpf_ext_signal_structural_break(ext);

    /* Next update should trigger sampling */
    rbpf_real_t ret = 0.05f * (rbpf_real_t)test_randn(); /* Bigger return */
    rbpf_ext_step(ext, ret, &output);

    uint64_t samples_after = ext->storvik.total_samples_drawn;
    uint64_t break_triggers_after = ext->storvik.samples_triggered_break;

    ASSERT(samples_after > samples_before, "Samples drawn after break signal");
    ASSERT(break_triggers_after > break_triggers_before, "Break triggers counted");

    rbpf_ext_destroy(ext);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 5: HFT MODE
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_hft_mode(void)
{
    printf("═══ Test 5: HFT Mode ═══\n");

    RBPF_Extended *ext = rbpf_ext_create(100, 4, RBPF_PARAM_STORVIK);
    rbpf_ext_init(ext, -4.6f, 0.1f);

    /* Check default intervals */
    ASSERT(ext->storvik.config.sample_interval[0] == 50, "Default R0 interval = 50");
    ASSERT(ext->storvik.config.sample_interval[3] == 1, "Default R3 interval = 1");

    /* Enable HFT mode */
    rbpf_ext_set_hft_mode(ext, 1);

    ASSERT(ext->storvik.config.sample_interval[0] == 100, "HFT R0 interval = 100");
    ASSERT(ext->storvik.config.sample_interval[3] == 5, "HFT R3 interval = 5");

    /* Disable HFT mode */
    rbpf_ext_set_hft_mode(ext, 0);

    ASSERT(ext->storvik.config.sample_interval[0] == 50, "Standard R0 interval restored");

    rbpf_ext_destroy(ext);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 6: Per-Particle Params Active in Storvik Mode (Option B Verification)
 *
 * Verifies that Option B is implemented correctly:
 *   - use_learned_params = 1 (predict reads particle arrays)
 *   - liu_west.enabled = 0 (Liu-West update logic does NOT run)
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_per_particle_params_active(void)
{
    printf("═══ Test 6: Per-Particle Params Active (Option B) ═══\n");

    RBPF_Extended *ext = rbpf_ext_create(100, 4, RBPF_PARAM_STORVIK);

    /* OPTION B CHECK: Decoupled flags */
    ASSERT(ext->rbpf->use_learned_params == 1,
           "use_learned_params == 1 (predict reads particle arrays)");
    ASSERT(ext->rbpf->liu_west.enabled == 0,
           "liu_west.enabled == 0 (no Liu-West update logic)");

    /* Verify Liu-West resample will NOT run */
    /* (It early-returns if liu_west.enabled == 0) */

    /* Set up filter */
    rbpf_ext_set_regime_params(ext, 0, 0.05f, -4.6f, 0.10f);
    rbpf_ext_set_regime_params(ext, 1, 0.08f, -3.5f, 0.15f);

    rbpf_real_t trans[16] = {
        0.95f, 0.05f, 0.00f, 0.00f,
        0.05f, 0.95f, 0.00f, 0.00f,
        0.00f, 0.00f, 1.00f, 0.00f,
        0.00f, 0.00f, 0.00f, 1.00f};
    rbpf_ext_build_transition_lut(ext, trans);
    rbpf_ext_init(ext, -4.6f, 0.1f);

    /* Manually set distinct per-particle mu_vol to verify they're used */
    int n = ext->rbpf->n_particles;
    int nr = ext->rbpf->n_regimes;
    for (int i = 0; i < n; i++)
    {
        for (int r = 0; r < nr; r++)
        {
            int idx = i * nr + r;
            /* Set to value different from global params */
            ext->rbpf->particle_mu_vol[idx] = -5.0f + 0.01f * i;
        }
    }

    /* Run predict step */
    rbpf_ksc_predict(ext->rbpf);

    /* Check that mu_pred reflects per-particle values, not global */
    /* If bug exists: mu_pred[0] ≈ (1-θ)*mu[0] + θ*(-4.6)  [global]
     * If fixed:      mu_pred[0] ≈ (1-θ)*mu[0] + θ*(-5.0)  [per-particle] */
    rbpf_real_t mu_pred_0 = ext->rbpf->mu_pred[0];
    rbpf_real_t mu_0 = ext->rbpf->mu[0];
    rbpf_real_t theta = ext->rbpf->params[0].theta;
    rbpf_real_t expected_global = (1.0f - theta) * mu_0 + theta * (-4.6f);
    rbpf_real_t expected_particle = (1.0f - theta) * mu_0 + theta * (-5.0f);

    /* mu_pred should be closer to expected_particle than expected_global */
    rbpf_real_t diff_global = fabsf(mu_pred_0 - expected_global);
    rbpf_real_t diff_particle = fabsf(mu_pred_0 - expected_particle);

    ASSERT(diff_particle < diff_global,
           "Predict uses per-particle mu_vol (not global params)");

    rbpf_ext_destroy(ext);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK 1: LATENCY COMPARISON
 *═══════════════════════════════════════════════════════════════════════════*/

static void bench_latency_comparison(void)
{
    printf("═══ Benchmark 1: Latency Comparison ═══\n");

    const int N_PARTICLES = 500;
    const int N_REGIMES = 4;
    const int N_ITERS = 2000;
    const int WARMUP = 200;

    /* Create three filters for comparison */
    RBPF_Extended *ext_lw = rbpf_ext_create(N_PARTICLES, N_REGIMES, RBPF_PARAM_LIU_WEST);
    RBPF_Extended *ext_sv = rbpf_ext_create(N_PARTICLES, N_REGIMES, RBPF_PARAM_STORVIK);
    RBPF_Extended *ext_hyb = rbpf_ext_create(N_PARTICLES, N_REGIMES, RBPF_PARAM_HYBRID);

    /* Configure all similarly */
    rbpf_real_t trans[16] = {
        0.95f, 0.04f, 0.01f, 0.00f,
        0.05f, 0.90f, 0.04f, 0.01f,
        0.01f, 0.05f, 0.90f, 0.04f,
        0.00f, 0.01f, 0.04f, 0.95f};

    rbpf_ext_build_transition_lut(ext_lw, trans);
    rbpf_ext_build_transition_lut(ext_sv, trans);
    rbpf_ext_build_transition_lut(ext_hyb, trans);

    rbpf_ext_init(ext_lw, -4.6f, 0.1f);
    rbpf_ext_init(ext_sv, -4.6f, 0.1f);
    rbpf_ext_init(ext_hyb, -4.6f, 0.1f);

    RBPF_KSC_Output output;
    Timer timer;

    /* Warmup */
    for (int i = 0; i < WARMUP; i++)
    {
        rbpf_real_t ret = 0.01f * (rbpf_real_t)test_randn();
        rbpf_ext_step(ext_lw, ret, &output);
        rbpf_ext_step(ext_sv, ret, &output);
        rbpf_ext_step(ext_hyb, ret, &output);
    }

    /* Benchmark Liu-West */
    double total_lw = 0;
    for (int i = 0; i < N_ITERS; i++)
    {
        rbpf_real_t ret = 0.01f * (rbpf_real_t)test_randn();
        timer_start(&timer);
        rbpf_ext_step(ext_lw, ret, &output);
        total_lw += timer_stop_us(&timer);
    }

    /* Benchmark Storvik */
    double total_sv = 0;
    for (int i = 0; i < N_ITERS; i++)
    {
        rbpf_real_t ret = 0.01f * (rbpf_real_t)test_randn();
        timer_start(&timer);
        rbpf_ext_step(ext_sv, ret, &output);
        total_sv += timer_stop_us(&timer);
    }

    /* Benchmark Hybrid */
    double total_hyb = 0;
    for (int i = 0; i < N_ITERS; i++)
    {
        rbpf_real_t ret = 0.01f * (rbpf_real_t)test_randn();
        timer_start(&timer);
        rbpf_ext_step(ext_hyb, ret, &output);
        total_hyb += timer_stop_us(&timer);
    }

    double avg_lw = total_lw / N_ITERS;
    double avg_sv = total_sv / N_ITERS;
    double avg_hyb = total_hyb / N_ITERS;

    printf("  Particles: %d\n", N_PARTICLES);
    printf("  Iterations: %d\n\n", N_ITERS);

    printf("  %-15s %10s %15s\n", "Mode", "Latency", "Overhead vs LW");
    printf("  %-15s %10s %15s\n", "----", "-------", "--------------");
    printf("  %-15s %8.2f μs %15s\n", "Liu-West", avg_lw, "(baseline)");
    printf("  %-15s %8.2f μs %+14.1f%%\n", "Storvik", avg_sv, 100.0 * (avg_sv - avg_lw) / avg_lw);
    printf("  %-15s %8.2f μs %+14.1f%%\n", "Hybrid", avg_hyb, 100.0 * (avg_hyb - avg_lw) / avg_lw);

    rbpf_ext_destroy(ext_lw);
    rbpf_ext_destroy(ext_sv);
    rbpf_ext_destroy(ext_hyb);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK 2: SLEEPING BEHAVIOR
 *═══════════════════════════════════════════════════════════════════════════*/

static void bench_sleeping_behavior(void)
{
    printf("═══ Benchmark 2: Sleeping Behavior ═══\n");

    const int N_PARTICLES = 500;
    const int N_ITERS = 1000;

    RBPF_Extended *ext = rbpf_ext_create(N_PARTICLES, 4, RBPF_PARAM_STORVIK);

    /* Set intervals: R0 sleeps, R3 awake */
    ext->storvik.config.sample_interval[0] = 100; /* R0: rare sampling */
    ext->storvik.config.sample_interval[1] = 50;
    ext->storvik.config.sample_interval[2] = 10;
    ext->storvik.config.sample_interval[3] = 1; /* R3: always sample */

    rbpf_real_t trans_r0[16] = {
        1.0f, 0.0f, 0.0f, 0.0f, /* Stay in R0 */
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f};
    rbpf_real_t trans_r3[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f /* Stay in R3 */
    };

    RBPF_KSC_Output output;
    Timer timer;

    /* Test R0 (sleeping) */
    rbpf_ext_build_transition_lut(ext, trans_r0);
    rbpf_ext_init(ext, -4.6f, 0.1f);

    /* Force all particles to R0 */
    for (int i = 0; i < N_PARTICLES; i++)
    {
        ext->rbpf->regime[i] = 0;
    }

    /* Warmup */
    for (int i = 0; i < 100; i++)
    {
        rbpf_ext_step(ext, 0.01f * (rbpf_real_t)test_randn(), &output);
    }

    double total_r0 = 0;
    uint64_t samples_r0_start = ext->storvik.total_samples_drawn;
    for (int i = 0; i < N_ITERS; i++)
    {
        timer_start(&timer);
        rbpf_ext_step(ext, 0.01f * (rbpf_real_t)test_randn(), &output);
        total_r0 += timer_stop_us(&timer);
    }
    uint64_t samples_r0 = ext->storvik.total_samples_drawn - samples_r0_start;

    /* Test R3 (awake) */
    rbpf_ext_build_transition_lut(ext, trans_r3);
    rbpf_ext_init(ext, -1.5f, 0.1f);

    /* Force all particles to R3 */
    for (int i = 0; i < N_PARTICLES; i++)
    {
        ext->rbpf->regime[i] = 3;
    }

    /* Warmup */
    for (int i = 0; i < 100; i++)
    {
        rbpf_ext_step(ext, 0.05f * (rbpf_real_t)test_randn(), &output);
    }

    double total_r3 = 0;
    uint64_t samples_r3_start = ext->storvik.total_samples_drawn;
    for (int i = 0; i < N_ITERS; i++)
    {
        timer_start(&timer);
        rbpf_ext_step(ext, 0.05f * (rbpf_real_t)test_randn(), &output);
        total_r3 += timer_stop_us(&timer);
    }
    uint64_t samples_r3 = ext->storvik.total_samples_drawn - samples_r3_start;

    double avg_r0 = total_r0 / N_ITERS;
    double avg_r3 = total_r3 / N_ITERS;

    printf("  Particles: %d, Iterations: %d\n\n", N_PARTICLES, N_ITERS);
    printf("  %-20s %10s %15s %15s\n", "Regime", "Latency", "Samples/iter", "Speedup");
    printf("  %-20s %10s %15s %15s\n", "------", "-------", "------------", "-------");
    printf("  %-20s %8.2f μs %15.1f %15s\n", "R0 (sleeping)", avg_r0,
           (double)samples_r0 / N_ITERS, "(baseline)");
    printf("  %-20s %8.2f μs %15.1f %14.1fx\n", "R3 (awake)", avg_r3,
           (double)samples_r3 / N_ITERS, avg_r3 / avg_r0);

    rbpf_ext_destroy(ext);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char **argv)
{
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║    RBPF-KSC + Param Learning Integration Test Suite          ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    /* Tests */
    test_initialization();
    test_regime_params_sync();
    test_basic_update();
    test_structural_break();
    test_hft_mode();
    test_per_particle_params_active(); /* Regression test for liu_west.enabled bug */

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Tests: %d passed, %d failed\n", g_passed, g_failed);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    if (g_failed > 0)
    {
        printf("  Some tests failed. Skipping benchmarks.\n\n");
        return 1;
    }

    /* Benchmarks */
    bench_latency_comparison();
    bench_sleeping_behavior();

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  All tests and benchmarks complete.\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    return 0;
}