/*=============================================================================
 * RBPF Comprehensive Comparison Test (v2)
 *
 * Compares 5 configurations:
 *   1. Baseline       - True params, no learning (oracle upper bound)
 *   2. Misspec        - Wrong params, no learning (realistic failure)
 *   3. Liu-West       - Legacy online learning
 *   4. Storvik        - Basic Storvik (no forgetting, no Robust OCSN)
 *   5. Storvik+Full   - Storvik + Adaptive Forgetting + Robust OCSN
 *
 * Scenarios tested (7500 ticks total):
 *   1. Calm period (R0) with occasional outliers
 *   2. Gradual volatility increase (R0 → R1 → R2)
 *   3. Sudden crisis spike (R2 → R3) with fat-tail events
 *   4. Crisis persistence with extreme moves
 *   5. Recovery (R3 → R2 → R1 → R0)
 *   6. Flash crash (brief R3 spike + 10σ outlier)
 *   7. Choppy regime switching
 *
 * Fat-tail injection:
 *   - Scenario 1: 2 outliers (6-8σ)
 *   - Scenario 3: 5 outliers (8-12σ) during crisis onset
 *   - Scenario 4: 3 outliers (10-15σ) extreme crisis
 *   - Scenario 6: 1 outlier (12σ) flash crash peak
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * BUILD
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Requires: rbpf_ksc.c, rbpf_param_learn.c, rbpf_ksc_param_integration.c
 *
 * Windows (MSVC + Intel MKL):
 *   cmake --build . --config Release --target test_rbpf_comparison
 *
 * Linux (GCC + Intel MKL):
 *   source /opt/intel/oneapi/setvars.sh
 *   make test_rbpf_comparison
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * USAGE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   test_rbpf_comparison [seed] [output_dir]
 *
 *   Examples:
 *     test_rbpf_comparison                    # seed=42, output to current dir
 *     test_rbpf_comparison 123 ./results      # seed=123, output to ./results/
 *
 *===========================================================================*/

#include "rbpf_ksc.h"
#include "rbpf_ksc_param_integration.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*─────────────────────────────────────────────────────────────────────────────
 * TIMING UTILITIES (Cross-platform)
 *───────────────────────────────────────────────────────────────────────────*/

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static double g_timer_freq = 0.0;

static void init_timer(void)
{
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    g_timer_freq = (double)freq.QuadPart / 1e6;
}

static inline double get_time_us(void)
{
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / g_timer_freq;
}
#else
#include <sys/time.h>

static void init_timer(void) {}

static inline double get_time_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}
#endif

/*─────────────────────────────────────────────────────────────────────────────
 * PCG32 RNG
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    uint64_t state;
    uint64_t inc;
} pcg32_t;

static uint32_t pcg32_random(pcg32_t *rng)
{
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static double pcg32_double(pcg32_t *rng)
{
    return (double)pcg32_random(rng) / 4294967296.0;
}

static double pcg32_gaussian(pcg32_t *rng)
{
    double u1 = pcg32_double(rng);
    double u2 = pcg32_double(rng);
    if (u1 < 1e-10)
        u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
}

/*─────────────────────────────────────────────────────────────────────────────
 * SYNTHETIC DATA GENERATION
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    rbpf_real_t *returns;
    rbpf_real_t *true_log_vol;
    rbpf_real_t *true_vol;
    int *true_regime;
    int *is_outlier;            /* Flag: 1 if this tick was an injected outlier */
    rbpf_real_t *outlier_sigma; /* How many sigma the outlier was */
    int n_ticks;
    int scenario_starts[10];
    int n_scenarios;
    int n_outliers_injected;
} SyntheticData;

/* Ground truth regime parameters */
static const rbpf_real_t TRUE_THETA[4] = {0.05f, 0.08f, 0.12f, 0.15f};
static const rbpf_real_t TRUE_MU_VOL[4] = {-4.6f, -3.5f, -2.5f, -1.5f};
static const rbpf_real_t TRUE_SIGMA_VOL[4] = {0.05f, 0.10f, 0.20f, 0.35f};

/* Inject an outlier: multiply return by factor to get N-sigma move */
static void inject_outlier(SyntheticData *data, int t, double target_sigma, pcg32_t *rng)
{
    /* Current volatility */
    rbpf_real_t vol = data->true_vol[t];

    /* Generate direction randomly */
    double sign = (pcg32_double(rng) < 0.5) ? -1.0 : 1.0;

    /* Set return to target_sigma * vol */
    data->returns[t] = (rbpf_real_t)(sign * target_sigma * vol);
    data->is_outlier[t] = 1;
    data->outlier_sigma[t] = (rbpf_real_t)target_sigma;
    data->n_outliers_injected++;
}

static SyntheticData *generate_test_data(int seed)
{
    SyntheticData *data = (SyntheticData *)calloc(1, sizeof(SyntheticData));

    int n = 7500;
    data->n_ticks = n;
    data->returns = (rbpf_real_t *)malloc(n * sizeof(rbpf_real_t));
    data->true_log_vol = (rbpf_real_t *)malloc(n * sizeof(rbpf_real_t));
    data->true_vol = (rbpf_real_t *)malloc(n * sizeof(rbpf_real_t));
    data->true_regime = (int *)malloc(n * sizeof(int));
    data->is_outlier = (int *)calloc(n, sizeof(int));
    data->outlier_sigma = (rbpf_real_t *)calloc(n, sizeof(rbpf_real_t));
    data->n_outliers_injected = 0;

    pcg32_t rng = {seed * 12345ULL + 1, seed * 67890ULL | 1};

    rbpf_real_t log_vol = TRUE_MU_VOL[0];
    int regime = 0;
    int t = 0;

/* Helper macro for state evolution */
#define EVOLVE_STATE(R)                                                                              \
    do                                                                                               \
    {                                                                                                \
        regime = (R);                                                                                \
        rbpf_real_t theta = TRUE_THETA[regime];                                                      \
        rbpf_real_t mu = TRUE_MU_VOL[regime];                                                        \
        rbpf_real_t sigma = TRUE_SIGMA_VOL[regime];                                                  \
        log_vol = (1.0f - theta) * log_vol + theta * mu + sigma * (rbpf_real_t)pcg32_gaussian(&rng); \
        rbpf_real_t vol = rbpf_exp(log_vol);                                                         \
        rbpf_real_t ret = vol * (rbpf_real_t)pcg32_gaussian(&rng);                                   \
        data->returns[t] = ret;                                                                      \
        data->true_log_vol[t] = log_vol;                                                             \
        data->true_vol[t] = vol;                                                                     \
        data->true_regime[t] = regime;                                                               \
    } while (0)

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 1: Calm (0-999) - R0 with 2 outliers
     *═══════════════════════════════════════════════════════════════════════*/
    data->scenario_starts[0] = 0;
    data->n_scenarios = 1;

    for (; t < 1000; t++)
    {
        EVOLVE_STATE(0);
    }

    /* Inject 2 outliers: 6σ and 8σ */
    inject_outlier(data, 300, 6.0, &rng);
    inject_outlier(data, 700, 8.0, &rng);

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 2: Gradual increase (1000-1999) - R0 → R1 → R2
     *═══════════════════════════════════════════════════════════════════════*/
    data->scenario_starts[1] = 1000;
    data->n_scenarios = 2;

    for (; t < 2000; t++)
    {
        int r;
        if (t < 1300)
            r = 0;
        else if (t < 1700)
            r = 1;
        else
            r = 2;
        EVOLVE_STATE(r);
    }

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 3: Sudden crisis (2000-2599) - Jump to R3 with fat tails
     *═══════════════════════════════════════════════════════════════════════*/
    data->scenario_starts[2] = 2000;
    data->n_scenarios = 3;

    for (; t < 2600; t++)
    {
        EVOLVE_STATE(3);
    }

    /* Inject 5 fat-tail events at crisis onset: 8-12σ */
    inject_outlier(data, 2010, 8.0, &rng);
    inject_outlier(data, 2025, 10.0, &rng);
    inject_outlier(data, 2050, 12.0, &rng);
    inject_outlier(data, 2100, 9.0, &rng);
    inject_outlier(data, 2200, 11.0, &rng);

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 4: Crisis persistence (2600-3799) - R2/R3 mix with extremes
     *═══════════════════════════════════════════════════════════════════════*/
    data->scenario_starts[3] = 2600;
    data->n_scenarios = 4;

    for (; t < 3800; t++)
    {
        /* Oscillate between R2 and R3 */
        if (pcg32_double(&rng) < 0.1 && regime == 3)
            regime = 2;
        else if (pcg32_double(&rng) < 0.2 && regime == 2)
            regime = 3;
        else if (regime < 2)
            regime = 3;

        EVOLVE_STATE(regime);
    }

    /* Inject 3 extreme outliers: 10-15σ */
    inject_outlier(data, 2800, 10.0, &rng);
    inject_outlier(data, 3200, 15.0, &rng); /* Extreme! */
    inject_outlier(data, 3500, 12.0, &rng);

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 5: Recovery (3800-4999) - R3 → R2 → R1 → R0
     *═══════════════════════════════════════════════════════════════════════*/
    data->scenario_starts[4] = 3800;
    data->n_scenarios = 5;

    for (; t < 5000; t++)
    {
        int r;
        if (t < 4100)
            r = 2;
        else if (t < 4500)
            r = 1;
        else
            r = 0;
        EVOLVE_STATE(r);
    }

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 6: Flash crash (5000-5499) - Brief R3 spike with 12σ outlier
     *═══════════════════════════════════════════════════════════════════════*/
    data->scenario_starts[5] = 5000;
    data->n_scenarios = 6;

    for (; t < 5500; t++)
    {
        int r;
        if (t >= 5200 && t < 5260)
            r = 3; /* 60-tick flash crash */
        else
            r = 0;
        EVOLVE_STATE(r);
    }

    /* Inject 12σ outlier at flash crash peak */
    inject_outlier(data, 5230, 12.0, &rng);

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 7: Choppy switching (5500-7499)
     *═══════════════════════════════════════════════════════════════════════*/
    data->scenario_starts[6] = 5500;
    data->n_scenarios = 7;

    int next_switch = 5500 + 50 + (int)(pcg32_double(&rng) * 100);
    regime = 1;

    for (; t < 7500; t++)
    {
        if (t >= next_switch)
        {
            int delta = (pcg32_double(&rng) < 0.5) ? -1 : 1;
            regime += delta;
            if (regime < 0)
                regime = 0;
            if (regime > 3)
                regime = 3;
            next_switch = t + 50 + (int)(pcg32_double(&rng) * 150);
        }
        EVOLVE_STATE(regime);
    }

#undef EVOLVE_STATE

    return data;
}

static void free_synthetic_data(SyntheticData *data)
{
    if (!data)
        return;
    free(data->returns);
    free(data->true_log_vol);
    free(data->true_vol);
    free(data->true_regime);
    free(data->is_outlier);
    free(data->outlier_sigma);
    free(data);
}

/*─────────────────────────────────────────────────────────────────────────────
 * TEST RECORD STRUCTURE
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    int tick;

    /* Ground truth */
    rbpf_real_t true_log_vol;
    rbpf_real_t true_vol;
    int true_regime;
    rbpf_real_t return_val;
    int is_outlier;
    rbpf_real_t outlier_sigma;

    /* Filter estimates */
    rbpf_real_t est_log_vol;
    rbpf_real_t est_vol;
    rbpf_real_t log_vol_var;
    rbpf_real_t ess;

    /* Regime */
    int est_regime;
    rbpf_real_t regime_prob[4];
    rbpf_real_t regime_entropy;

    /* Learned parameters */
    rbpf_real_t learned_mu_vol[4];
    rbpf_real_t learned_sigma_vol[4];

    /* Robust OCSN */
    rbpf_real_t outlier_fraction; /* Posterior prob of outlier component */

    /* Detection */
    rbpf_real_t surprise;
    rbpf_real_t vol_ratio;
    int regime_changed;

    /* Timing */
    double latency_us;
    int resampled;

} TickRecord;

/*─────────────────────────────────────────────────────────────────────────────
 * TEST MODES
 *───────────────────────────────────────────────────────────────────────────*/

typedef enum
{
    MODE_BASELINE = 0,   /* RBPF, true params, no learning */
    MODE_MISSPEC,        /* RBPF, wrong params, no learning */
    MODE_LIU_WEST,       /* RBPF + Liu-West learning */
    MODE_STORVIK,        /* RBPF + Storvik (no forgetting, no OCSN) */
    MODE_STORVIK_FORGET, /* RBPF + Storvik + Forgetting (no OCSN) */
    MODE_STORVIK_FULL,   /* RBPF + Storvik + Forgetting + Robust OCSN */
    NUM_MODES
} TestMode;

static const char *mode_names[] = {
    "Baseline",
    "Misspec",
    "Liu-West",
    "Storvik",
    "Storvik+Fgt",
    "Storvik+Full"};

static const char *csv_names[] = {
    "rbpf_baseline.csv",
    "rbpf_misspec.csv",
    "rbpf_liu_west.csv",
    "rbpf_storvik.csv",
    "rbpf_storvik_forget.csv",
    "rbpf_storvik_full.csv"};

/*─────────────────────────────────────────────────────────────────────────────
 * RUN TEST
 *───────────────────────────────────────────────────────────────────────────*/

static void run_test(SyntheticData *data, TestMode mode, TickRecord *records,
                     double *total_time_us, double *max_latency_us)
{
    const int N_PARTICLES = 512;
    const int N_REGIMES = 4;

    RBPF_Extended *ext = NULL;
    RBPF_KSC *rbpf_raw = NULL;

    /* Transition matrix */
    rbpf_real_t trans[16] = {
        0.95f, 0.04f, 0.01f, 0.00f,
        0.04f, 0.90f, 0.05f, 0.01f,
        0.01f, 0.05f, 0.90f, 0.04f,
        0.00f, 0.01f, 0.04f, 0.95f};

    /* Create filter based on mode */
    switch (mode)
    {
    case MODE_BASELINE:
        rbpf_raw = rbpf_ksc_create(N_PARTICLES, N_REGIMES);
        for (int r = 0; r < N_REGIMES; r++)
        {
            rbpf_ksc_set_regime_params(rbpf_raw, r, TRUE_THETA[r], TRUE_MU_VOL[r], TRUE_SIGMA_VOL[r]);
        }
        rbpf_ksc_build_transition_lut(rbpf_raw, trans);
        rbpf_ksc_init(rbpf_raw, TRUE_MU_VOL[0], 0.1f);
        break;

    case MODE_MISSPEC:
        /*
         * Realistic scenario: parameters are wrong
         *   mu_vol:    20% too high (closer to 0 = less volatile)
         *   sigma_vol: 30% too low (underestimate noise)
         */
        rbpf_raw = rbpf_ksc_create(N_PARTICLES, N_REGIMES);
        for (int r = 0; r < N_REGIMES; r++)
        {
            rbpf_real_t wrong_mu = TRUE_MU_VOL[r] * 0.8f;
            rbpf_real_t wrong_sigma = TRUE_SIGMA_VOL[r] * 0.7f;
            rbpf_ksc_set_regime_params(rbpf_raw, r, TRUE_THETA[r], wrong_mu, wrong_sigma);
        }
        rbpf_ksc_build_transition_lut(rbpf_raw, trans);
        rbpf_ksc_init(rbpf_raw, TRUE_MU_VOL[0] * 0.8f, 0.1f);
        break;

    case MODE_LIU_WEST:
        rbpf_raw = rbpf_ksc_create(N_PARTICLES, N_REGIMES);
        for (int r = 0; r < N_REGIMES; r++)
        {
            rbpf_ksc_set_regime_params(rbpf_raw, r, TRUE_THETA[r], TRUE_MU_VOL[r], TRUE_SIGMA_VOL[r]);
        }
        rbpf_ksc_build_transition_lut(rbpf_raw, trans);
        rbpf_ksc_enable_liu_west(rbpf_raw, 0.98f, 100);
        rbpf_ksc_init(rbpf_raw, TRUE_MU_VOL[0], 0.1f);
        break;

    case MODE_STORVIK:
        /* Basic Storvik: no forgetting, no Robust OCSN */
        ext = rbpf_ext_create(N_PARTICLES, N_REGIMES, RBPF_PARAM_STORVIK);
        for (int r = 0; r < N_REGIMES; r++)
        {
            rbpf_ext_set_regime_params(ext, r, TRUE_THETA[r], TRUE_MU_VOL[r], TRUE_SIGMA_VOL[r]);
        }
        rbpf_ext_build_transition_lut(ext, trans);

        /* Disable forgetting */
        param_learn_set_forgetting(&ext->storvik, 0, 1.0f); /* lambda=1.0 = no forgetting */

        rbpf_ext_init(ext, TRUE_MU_VOL[0], 0.1f);
        break;

    case MODE_STORVIK_FORGET:
        /* Storvik + Adaptive Forgetting (no Robust OCSN) */
        ext = rbpf_ext_create(N_PARTICLES, N_REGIMES, RBPF_PARAM_STORVIK);
        for (int r = 0; r < N_REGIMES; r++)
        {
            rbpf_ext_set_regime_params(ext, r, TRUE_THETA[r], TRUE_MU_VOL[r], TRUE_SIGMA_VOL[r]);
        }
        rbpf_ext_build_transition_lut(ext, trans);

        /* Enable regime-adaptive forgetting */
        param_learn_set_forgetting(&ext->storvik, 1, 0.997f);        /* Default λ */
        param_learn_set_regime_forgetting(&ext->storvik, 0, 0.999f); /* R0: slow (N_eff≈1000) */
        param_learn_set_regime_forgetting(&ext->storvik, 1, 0.998f); /* R1 */
        param_learn_set_regime_forgetting(&ext->storvik, 2, 0.996f); /* R2 */
        param_learn_set_regime_forgetting(&ext->storvik, 3, 0.993f); /* R3: fast (N_eff≈143) */

        /* Robust OCSN DISABLED */
        ext->robust_ocsn.enabled = 0;

        rbpf_ext_init(ext, TRUE_MU_VOL[0], 0.1f);
        break;

    case MODE_STORVIK_FULL:
        /* Full stack: Storvik + Adaptive Forgetting + Robust OCSN */
        ext = rbpf_ext_create(N_PARTICLES, N_REGIMES, RBPF_PARAM_STORVIK);
        for (int r = 0; r < N_REGIMES; r++)
        {
            rbpf_ext_set_regime_params(ext, r, TRUE_THETA[r], TRUE_MU_VOL[r], TRUE_SIGMA_VOL[r]);
        }
        rbpf_ext_build_transition_lut(ext, trans);

        /* Enable regime-adaptive forgetting */
        param_learn_set_forgetting(&ext->storvik, 1, 0.997f);        /* Default λ */
        param_learn_set_regime_forgetting(&ext->storvik, 0, 0.999f); /* R0: slow (N_eff≈1000) */
        param_learn_set_regime_forgetting(&ext->storvik, 1, 0.998f); /* R1 */
        param_learn_set_regime_forgetting(&ext->storvik, 2, 0.996f); /* R2 */
        param_learn_set_regime_forgetting(&ext->storvik, 3, 0.993f); /* R3: fast (N_eff≈143) */

        /* Enable Robust OCSN with per-regime outlier params */
        ext->robust_ocsn.enabled = 1;
        ext->robust_ocsn.regime[0].prob = 0.010f; /* R0: 1.0% outlier prob */
        ext->robust_ocsn.regime[0].variance = 18.0f;
        ext->robust_ocsn.regime[1].prob = 0.015f; /* R1: 1.5% */
        ext->robust_ocsn.regime[1].variance = 22.0f;
        ext->robust_ocsn.regime[2].prob = 0.020f; /* R2: 2.0% */
        ext->robust_ocsn.regime[2].variance = 26.0f;
        ext->robust_ocsn.regime[3].prob = 0.025f; /* R3: 2.5% */
        ext->robust_ocsn.regime[3].variance = 30.0f;

        rbpf_ext_init(ext, TRUE_MU_VOL[0], 0.1f);
        break;
    }

    *total_time_us = 0.0;
    *max_latency_us = 0.0;

    int n = data->n_ticks;
    RBPF_KSC_Output output;

    for (int t = 0; t < n; t++)
    {
        memset(&output, 0, sizeof(output));
        double t_start = get_time_us();

        rbpf_real_t ret = data->returns[t];

        switch (mode)
        {
        case MODE_BASELINE:
        case MODE_MISSPEC:
        case MODE_LIU_WEST:
            rbpf_ksc_step(rbpf_raw, ret, &output);
            break;

        case MODE_STORVIK:
        case MODE_STORVIK_FORGET:
        case MODE_STORVIK_FULL:
            rbpf_ext_step(ext, ret, &output);
            break;
        }

        double t_end = get_time_us();
        double latency = t_end - t_start;

        *total_time_us += latency;
        if (latency > *max_latency_us)
            *max_latency_us = latency;

        /* Record results */
        TickRecord *rec = &records[t];
        rec->tick = t;

        /* Ground truth */
        rec->true_log_vol = data->true_log_vol[t];
        rec->true_vol = data->true_vol[t];
        rec->true_regime = data->true_regime[t];
        rec->return_val = ret;
        rec->is_outlier = data->is_outlier[t];
        rec->outlier_sigma = data->outlier_sigma[t];

        /* Filter estimates */
        rec->est_log_vol = output.log_vol_mean;
        rec->est_vol = output.vol_mean;
        rec->log_vol_var = output.log_vol_var;
        rec->ess = output.ess;

        /* Regime */
        rec->est_regime = output.dominant_regime;
        for (int r = 0; r < 4; r++)
        {
            rec->regime_prob[r] = output.regime_probs[r];
            rec->learned_mu_vol[r] = output.learned_mu_vol[r];
            rec->learned_sigma_vol[r] = output.learned_sigma_vol[r];
        }
        rec->regime_entropy = output.regime_entropy;

        /* Robust OCSN */
        rec->outlier_fraction = output.outlier_fraction;

        /* Detection */
        rec->surprise = output.surprise;
        rec->vol_ratio = output.vol_ratio;
        rec->regime_changed = output.regime_changed;

        /* Timing */
        rec->latency_us = latency;
        rec->resampled = output.resampled;
    }

    /* Cleanup */
    if (rbpf_raw)
        rbpf_ksc_destroy(rbpf_raw);
    if (ext)
        rbpf_ext_destroy(ext);
}

/*─────────────────────────────────────────────────────────────────────────────
 * WRITE CSV
 *───────────────────────────────────────────────────────────────────────────*/

static void write_tick_csv(const char *filename, TickRecord *records, int n)
{
    FILE *f = fopen(filename, "w");
    if (!f)
    {
        fprintf(stderr, "Failed to open %s\n", filename);
        return;
    }

    /* Header */
    fprintf(f, "tick,true_log_vol,true_vol,true_regime,return,is_outlier,outlier_sigma,"
               "est_log_vol,est_vol,log_vol_var,ess,"
               "est_regime,regime_prob_0,regime_prob_1,regime_prob_2,regime_prob_3,regime_entropy,"
               "learned_mu_r0,learned_sigma_r0,learned_mu_r1,learned_sigma_r1,"
               "learned_mu_r2,learned_sigma_r2,learned_mu_r3,learned_sigma_r3,"
               "outlier_fraction,surprise,vol_ratio,regime_changed,latency_us,resampled\n");

    for (int t = 0; t < n; t++)
    {
        TickRecord *r = &records[t];
        fprintf(f, "%d,%.6f,%.6f,%d,%.8f,%d,%.1f,"
                   "%.6f,%.6f,%.6f,%.2f,"
                   "%d,%.4f,%.4f,%.4f,%.4f,%.4f,"
                   "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                   "%.4f,%.4f,%.4f,%d,%.2f,%d\n",
                r->tick, r->true_log_vol, r->true_vol, r->true_regime, r->return_val,
                r->is_outlier, r->outlier_sigma,
                r->est_log_vol, r->est_vol, r->log_vol_var, r->ess,
                r->est_regime, r->regime_prob[0], r->regime_prob[1], r->regime_prob[2], r->regime_prob[3],
                r->regime_entropy,
                r->learned_mu_vol[0], r->learned_sigma_vol[0],
                r->learned_mu_vol[1], r->learned_sigma_vol[1],
                r->learned_mu_vol[2], r->learned_sigma_vol[2],
                r->learned_mu_vol[3], r->learned_sigma_vol[3],
                r->outlier_fraction, r->surprise, r->vol_ratio, r->regime_changed,
                r->latency_us, r->resampled);
    }

    fclose(f);
    printf("  Written: %s (%d rows)\n", filename, n);
}

/*─────────────────────────────────────────────────────────────────────────────
 * COMPUTE METRICS
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    /* State estimation */
    double log_vol_rmse;
    double log_vol_mae;
    double vol_rmse;
    double vol_mae;

    /* Regime detection */
    double regime_accuracy;

    /* Particle filter health */
    double avg_ess;
    double min_ess;
    int resample_count;
    int ess_collapse_count; /* ESS < 10% of N */

    /* Timing */
    double avg_latency_us;
    double max_latency_us;
    double p99_latency_us;

    /* Parameter learning (error vs truth at end) */
    double mu_vol_error[4];
    double sigma_vol_error[4];

    /* Outlier handling */
    double avg_outlier_fraction_on_outliers; /* When is_outlier=1 */
    double avg_outlier_fraction_on_normal;   /* When is_outlier=0 */
    int outlier_detection_count;             /* outlier_fraction > 0.5 when is_outlier=1 */
    int false_outlier_count;                 /* outlier_fraction > 0.5 when is_outlier=0 */

    /* Robustness: error on outlier ticks vs normal ticks */
    double log_vol_rmse_on_outliers;
    double log_vol_rmse_on_normal;

} SummaryMetrics;

static int compare_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static void compute_summary(TickRecord *records, SyntheticData *data,
                            SummaryMetrics *m, int n_particles)
{
    int n = data->n_ticks;
    memset(m, 0, sizeof(SummaryMetrics));

    double sum_log_err2 = 0, sum_log_err = 0;
    double sum_vol_err2 = 0, sum_vol_err = 0;
    int regime_correct = 0;
    double sum_ess = 0;
    double min_ess = 1e9;
    double sum_latency = 0;
    double max_latency = 0;

    /* Outlier-specific accumulators */
    double sum_log_err2_outlier = 0;
    double sum_log_err2_normal = 0;
    int n_outlier = 0, n_normal = 0;
    double sum_outlier_frac_on_outlier = 0;
    double sum_outlier_frac_on_normal = 0;

    double ess_threshold = 0.1 * n_particles;

    double *latencies = (double *)malloc(n * sizeof(double));

    for (int t = 0; t < n; t++)
    {
        TickRecord *r = &records[t];

        double log_err = r->est_log_vol - r->true_log_vol;
        double vol_err = r->est_vol - r->true_vol;

        sum_log_err += fabs(log_err);
        sum_log_err2 += log_err * log_err;
        sum_vol_err += fabs(vol_err);
        sum_vol_err2 += vol_err * vol_err;

        if (r->est_regime == r->true_regime)
            regime_correct++;

        sum_ess += r->ess;
        if (r->ess < min_ess)
            min_ess = r->ess;
        if (r->ess < ess_threshold)
            m->ess_collapse_count++;
        if (r->resampled)
            m->resample_count++;

        latencies[t] = r->latency_us;
        sum_latency += r->latency_us;
        if (r->latency_us > max_latency)
            max_latency = r->latency_us;

        /* Outlier-specific metrics */
        if (data->is_outlier[t])
        {
            sum_log_err2_outlier += log_err * log_err;
            sum_outlier_frac_on_outlier += r->outlier_fraction;
            n_outlier++;
            if (r->outlier_fraction > 0.5f)
                m->outlier_detection_count++;
        }
        else
        {
            sum_log_err2_normal += log_err * log_err;
            sum_outlier_frac_on_normal += r->outlier_fraction;
            n_normal++;
            if (r->outlier_fraction > 0.5f)
                m->false_outlier_count++;
        }
    }

    m->log_vol_rmse = sqrt(sum_log_err2 / n);
    m->log_vol_mae = sum_log_err / n;
    m->vol_rmse = sqrt(sum_vol_err2 / n);
    m->vol_mae = sum_vol_err / n;
    m->regime_accuracy = (double)regime_correct / n;
    m->avg_latency_us = sum_latency / n;
    m->max_latency_us = max_latency;
    m->avg_ess = sum_ess / n;
    m->min_ess = min_ess;

    /* Outlier metrics */
    if (n_outlier > 0)
    {
        m->log_vol_rmse_on_outliers = sqrt(sum_log_err2_outlier / n_outlier);
        m->avg_outlier_fraction_on_outliers = sum_outlier_frac_on_outlier / n_outlier;
    }
    if (n_normal > 0)
    {
        m->log_vol_rmse_on_normal = sqrt(sum_log_err2_normal / n_normal);
        m->avg_outlier_fraction_on_normal = sum_outlier_frac_on_normal / n_normal;
    }

    /* P99 latency */
    qsort(latencies, n, sizeof(double), compare_double);
    m->p99_latency_us = latencies[(int)(0.99 * n)];
    free(latencies);

    /* Parameter learning error at end */
    TickRecord *last = &records[n - 1];
    for (int r = 0; r < 4; r++)
    {
        m->mu_vol_error[r] = fabs(last->learned_mu_vol[r] - TRUE_MU_VOL[r]);
        m->sigma_vol_error[r] = fabs(last->learned_sigma_vol[r] - TRUE_SIGMA_VOL[r]);
    }
}

static void write_summary_csv(const char *filename, SummaryMetrics *metrics)
{
    FILE *f = fopen(filename, "w");
    if (!f)
    {
        fprintf(stderr, "Failed to open %s\n", filename);
        return;
    }

    fprintf(f, "metric,baseline,misspec,liu_west,storvik,storvik_forget,storvik_full\n");

    fprintf(f, "log_vol_rmse,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            metrics[0].log_vol_rmse, metrics[1].log_vol_rmse,
            metrics[2].log_vol_rmse, metrics[3].log_vol_rmse,
            metrics[4].log_vol_rmse, metrics[5].log_vol_rmse);
    fprintf(f, "log_vol_mae,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            metrics[0].log_vol_mae, metrics[1].log_vol_mae,
            metrics[2].log_vol_mae, metrics[3].log_vol_mae,
            metrics[4].log_vol_mae, metrics[5].log_vol_mae);
    fprintf(f, "vol_rmse,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            metrics[0].vol_rmse, metrics[1].vol_rmse,
            metrics[2].vol_rmse, metrics[3].vol_rmse,
            metrics[4].vol_rmse, metrics[5].vol_rmse);
    fprintf(f, "regime_accuracy,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            metrics[0].regime_accuracy, metrics[1].regime_accuracy,
            metrics[2].regime_accuracy, metrics[3].regime_accuracy,
            metrics[4].regime_accuracy, metrics[5].regime_accuracy);
    fprintf(f, "avg_ess,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
            metrics[0].avg_ess, metrics[1].avg_ess,
            metrics[2].avg_ess, metrics[3].avg_ess,
            metrics[4].avg_ess, metrics[5].avg_ess);
    fprintf(f, "min_ess,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
            metrics[0].min_ess, metrics[1].min_ess,
            metrics[2].min_ess, metrics[3].min_ess,
            metrics[4].min_ess, metrics[5].min_ess);
    fprintf(f, "ess_collapse_count,%d,%d,%d,%d,%d,%d\n",
            metrics[0].ess_collapse_count, metrics[1].ess_collapse_count,
            metrics[2].ess_collapse_count, metrics[3].ess_collapse_count,
            metrics[4].ess_collapse_count, metrics[5].ess_collapse_count);
    fprintf(f, "avg_latency_us,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
            metrics[0].avg_latency_us, metrics[1].avg_latency_us,
            metrics[2].avg_latency_us, metrics[3].avg_latency_us,
            metrics[4].avg_latency_us, metrics[5].avg_latency_us);
    fprintf(f, "p99_latency_us,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
            metrics[0].p99_latency_us, metrics[1].p99_latency_us,
            metrics[2].p99_latency_us, metrics[3].p99_latency_us,
            metrics[4].p99_latency_us, metrics[5].p99_latency_us);
    fprintf(f, "mu_vol_r0_error,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            metrics[0].mu_vol_error[0], metrics[1].mu_vol_error[0],
            metrics[2].mu_vol_error[0], metrics[3].mu_vol_error[0],
            metrics[4].mu_vol_error[0], metrics[5].mu_vol_error[0]);
    fprintf(f, "mu_vol_r3_error,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            metrics[0].mu_vol_error[3], metrics[1].mu_vol_error[3],
            metrics[2].mu_vol_error[3], metrics[3].mu_vol_error[3],
            metrics[4].mu_vol_error[3], metrics[5].mu_vol_error[3]);
    fprintf(f, "log_vol_rmse_on_outliers,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            metrics[0].log_vol_rmse_on_outliers, metrics[1].log_vol_rmse_on_outliers,
            metrics[2].log_vol_rmse_on_outliers, metrics[3].log_vol_rmse_on_outliers,
            metrics[4].log_vol_rmse_on_outliers, metrics[5].log_vol_rmse_on_outliers);
    fprintf(f, "log_vol_rmse_on_normal,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            metrics[0].log_vol_rmse_on_normal, metrics[1].log_vol_rmse_on_normal,
            metrics[2].log_vol_rmse_on_normal, metrics[3].log_vol_rmse_on_normal,
            metrics[4].log_vol_rmse_on_normal, metrics[5].log_vol_rmse_on_normal);
    fprintf(f, "avg_outlier_frac_on_outliers,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            metrics[0].avg_outlier_fraction_on_outliers, metrics[1].avg_outlier_fraction_on_outliers,
            metrics[2].avg_outlier_fraction_on_outliers, metrics[3].avg_outlier_fraction_on_outliers,
            metrics[4].avg_outlier_fraction_on_outliers, metrics[5].avg_outlier_fraction_on_outliers);
    fprintf(f, "outlier_detection_count,%d,%d,%d,%d,%d,%d\n",
            metrics[0].outlier_detection_count, metrics[1].outlier_detection_count,
            metrics[2].outlier_detection_count, metrics[3].outlier_detection_count,
            metrics[4].outlier_detection_count, metrics[5].outlier_detection_count);

    fclose(f);
    printf("  Written: %s\n", filename);
}

/*─────────────────────────────────────────────────────────────────────────────
 * PRINT SUMMARY TABLE
 *───────────────────────────────────────────────────────────────────────────*/

static void print_summary_table(SummaryMetrics *metrics, SyntheticData *data,
                                TickRecord *records[NUM_MODES])
{
    int n = data->n_ticks;

    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  SUMMARY COMPARISON (N=512 particles, %d ticks, %d injected outliers)\n",
           n, data->n_outliers_injected);
    printf("══════════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("%-24s %10s %10s %10s %10s %12s %12s\n",
           "Metric", "Baseline", "Misspec", "Liu-West", "Storvik", "Storvik+Fgt", "Storvik+Full");
    printf("──────────────────────────────────────────────────────────────────────────────────────────────────\n");

    /* State estimation */
    printf("%-24s %10.4f %10.4f %10.4f %10.4f %12.4f %12.4f\n", "Log-Vol RMSE",
           metrics[0].log_vol_rmse, metrics[1].log_vol_rmse,
           metrics[2].log_vol_rmse, metrics[3].log_vol_rmse,
           metrics[4].log_vol_rmse, metrics[5].log_vol_rmse);
    printf("%-24s %10.4f %10.4f %10.4f %10.4f %12.4f %12.4f\n", "Vol RMSE",
           metrics[0].vol_rmse, metrics[1].vol_rmse,
           metrics[2].vol_rmse, metrics[3].vol_rmse,
           metrics[4].vol_rmse, metrics[5].vol_rmse);

    /* Regime accuracy */
    printf("%-24s %9.1f%% %9.1f%% %9.1f%% %9.1f%% %11.1f%% %11.1f%%\n", "Regime Accuracy",
           100 * metrics[0].regime_accuracy, 100 * metrics[1].regime_accuracy,
           100 * metrics[2].regime_accuracy, 100 * metrics[3].regime_accuracy,
           100 * metrics[4].regime_accuracy, 100 * metrics[5].regime_accuracy);

    /* ESS health */
    printf("%-24s %10.1f %10.1f %10.1f %10.1f %12.1f %12.1f\n", "Avg ESS",
           metrics[0].avg_ess, metrics[1].avg_ess,
           metrics[2].avg_ess, metrics[3].avg_ess,
           metrics[4].avg_ess, metrics[5].avg_ess);
    printf("%-24s %10.1f %10.1f %10.1f %10.1f %12.1f %12.1f\n", "Min ESS",
           metrics[0].min_ess, metrics[1].min_ess,
           metrics[2].min_ess, metrics[3].min_ess,
           metrics[4].min_ess, metrics[5].min_ess);
    printf("%-24s %10d %10d %10d %10d %12d %12d\n", "ESS Collapse Count",
           metrics[0].ess_collapse_count, metrics[1].ess_collapse_count,
           metrics[2].ess_collapse_count, metrics[3].ess_collapse_count,
           metrics[4].ess_collapse_count, metrics[5].ess_collapse_count);

    /* Timing */
    printf("%-24s %10.2f %10.2f %10.2f %10.2f %12.2f %12.2f\n", "Avg Latency (us)",
           metrics[0].avg_latency_us, metrics[1].avg_latency_us,
           metrics[2].avg_latency_us, metrics[3].avg_latency_us,
           metrics[4].avg_latency_us, metrics[5].avg_latency_us);
    printf("%-24s %10.2f %10.2f %10.2f %10.2f %12.2f %12.2f\n", "P99 Latency (us)",
           metrics[0].p99_latency_us, metrics[1].p99_latency_us,
           metrics[2].p99_latency_us, metrics[3].p99_latency_us,
           metrics[4].p99_latency_us, metrics[5].p99_latency_us);

    printf("──────────────────────────────────────────────────────────────────────────────────────────────────\n");
    printf("PARAMETER LEARNING (Error vs Truth at End)\n");
    printf("──────────────────────────────────────────────────────────────────────────────────────────────────\n");
    printf("%-24s %10.4f %10.4f %10.4f %10.4f %12.4f %12.4f\n", "mu_vol R0 Error",
           metrics[0].mu_vol_error[0], metrics[1].mu_vol_error[0],
           metrics[2].mu_vol_error[0], metrics[3].mu_vol_error[0],
           metrics[4].mu_vol_error[0], metrics[5].mu_vol_error[0]);
    printf("%-24s %10.4f %10.4f %10.4f %10.4f %12.4f %12.4f\n", "mu_vol R3 Error",
           metrics[0].mu_vol_error[3], metrics[1].mu_vol_error[3],
           metrics[2].mu_vol_error[3], metrics[3].mu_vol_error[3],
           metrics[4].mu_vol_error[3], metrics[5].mu_vol_error[3]);

    printf("──────────────────────────────────────────────────────────────────────────────────────────────────\n");
    printf("OUTLIER ROBUSTNESS (%d injected outliers, 6-15σ)\n", data->n_outliers_injected);
    printf("──────────────────────────────────────────────────────────────────────────────────────────────────\n");
    printf("%-24s %10.4f %10.4f %10.4f %10.4f %12.4f %12.4f\n", "RMSE on Outlier Ticks",
           metrics[0].log_vol_rmse_on_outliers, metrics[1].log_vol_rmse_on_outliers,
           metrics[2].log_vol_rmse_on_outliers, metrics[3].log_vol_rmse_on_outliers,
           metrics[4].log_vol_rmse_on_outliers, metrics[5].log_vol_rmse_on_outliers);
    printf("%-24s %10.4f %10.4f %10.4f %10.4f %12.4f %12.4f\n", "RMSE on Normal Ticks",
           metrics[0].log_vol_rmse_on_normal, metrics[1].log_vol_rmse_on_normal,
           metrics[2].log_vol_rmse_on_normal, metrics[3].log_vol_rmse_on_normal,
           metrics[4].log_vol_rmse_on_normal, metrics[5].log_vol_rmse_on_normal);
    printf("%-24s %10.2f %10.2f %10.2f %10.2f %12.2f %12.2f\n", "Avg Outlier Frac (outliers)",
           metrics[0].avg_outlier_fraction_on_outliers, metrics[1].avg_outlier_fraction_on_outliers,
           metrics[2].avg_outlier_fraction_on_outliers, metrics[3].avg_outlier_fraction_on_outliers,
           metrics[4].avg_outlier_fraction_on_outliers, metrics[5].avg_outlier_fraction_on_outliers);
    printf("%-24s %10d %10d %10d %10d %12d %12d\n", "Outlier Detections",
           metrics[0].outlier_detection_count, metrics[1].outlier_detection_count,
           metrics[2].outlier_detection_count, metrics[3].outlier_detection_count,
           metrics[4].outlier_detection_count, metrics[5].outlier_detection_count);
    printf("══════════════════════════════════════════════════════════════════════════════════════════════════\n");

    /* Scenario breakdown */
    printf("\nSCENARIO BREAKDOWN (Regime Accuracy %%)\n");
    printf("──────────────────────────────────────────────────────────────────────────────────────────────────\n");

    const char *scenario_names[] = {
        "1. Calm + 2 outliers",
        "2. Gradual Rise",
        "3. Crisis + 5 fat-tails",
        "4. Crisis Persist + 3 ext",
        "5. Recovery",
        "6. Flash Crash + 12σ",
        "7. Choppy"};

    printf("%-24s %10s %10s %10s %10s %12s %12s\n",
           "Scenario", "Baseline", "Misspec", "Liu-West", "Storvik", "Storvik+Fgt", "Storvik+Full");
    printf("──────────────────────────────────────────────────────────────────────────────────────────────────\n");

    for (int s = 0; s < data->n_scenarios; s++)
    {
        int start = data->scenario_starts[s];
        int end = (s + 1 < data->n_scenarios) ? data->scenario_starts[s + 1] : n;
        int count = end - start;

        int correct[NUM_MODES] = {0};
        for (int t = start; t < end; t++)
        {
            for (int m = 0; m < NUM_MODES; m++)
            {
                if (records[m][t].est_regime == data->true_regime[t])
                    correct[m]++;
            }
        }

        printf("%-24s %9.1f%% %9.1f%% %9.1f%% %9.1f%% %11.1f%% %11.1f%%\n", scenario_names[s],
               100.0 * correct[0] / count,
               100.0 * correct[1] / count,
               100.0 * correct[2] / count,
               100.0 * correct[3] / count,
               100.0 * correct[4] / count,
               100.0 * correct[5] / count);
    }
    printf("══════════════════════════════════════════════════════════════════════════════════════════════════\n");
}

/*─────────────────────────────────────────────────────────────────────────────
 * MAIN
 *───────────────────────────────────────────────────────────────────────────*/

int main(int argc, char **argv)
{
    int seed = 42;
    const char *output_dir = ".";

    if (argc > 1)
        seed = atoi(argv[1]);
    if (argc > 2)
        output_dir = argv[2];

    init_timer();

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  RBPF Comprehensive Comparison Test v2\n");
    printf("  (with Robust OCSN + Adaptive Forgetting)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Seed: %d\n", seed);
    printf("  Output: %s\n\n", output_dir);

    /* Generate data */
    printf("Generating synthetic data with fat-tail injection...\n");
    SyntheticData *data = generate_test_data(seed);
    printf("  Ticks: %d\n", data->n_ticks);
    printf("  Scenarios: %d\n", data->n_scenarios);
    printf("  Injected outliers: %d (6-15σ)\n\n", data->n_outliers_injected);

    int n = data->n_ticks;
    TickRecord *records[NUM_MODES];
    SummaryMetrics metrics[NUM_MODES];
    double total_time[NUM_MODES], max_latency[NUM_MODES];

    /* Run all modes */
    for (int mode = 0; mode < NUM_MODES; mode++)
    {
        printf("Running %s...\n", mode_names[mode]);
        records[mode] = (TickRecord *)calloc(n, sizeof(TickRecord));

        run_test(data, (TestMode)mode, records[mode], &total_time[mode], &max_latency[mode]);
        compute_summary(records[mode], data, &metrics[mode], 512);

        printf("  Total: %.2f ms, Avg: %.2f us, Max: %.2f us\n\n",
               total_time[mode] / 1000.0,
               total_time[mode] / n,
               max_latency[mode]);
    }

    /* Write CSVs */
    printf("Writing CSV files to: %s\n", output_dir);

    for (int mode = 0; mode < NUM_MODES; mode++)
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", output_dir, csv_names[mode]);
        write_tick_csv(path, records[mode], n);
    }

    char summary_path[512];
    snprintf(summary_path, sizeof(summary_path), "%s/rbpf_comparison_summary.csv", output_dir);
    write_summary_csv(summary_path, metrics);

    /* Print summary table */
    print_summary_table(metrics, data, records);

    /* Cleanup */
    for (int m = 0; m < NUM_MODES; m++)
        free(records[m]);
    free_synthetic_data(data);

    printf("\nDone. Open CSV files for detailed analysis.\n");
    printf("Key metrics to compare:\n");
    printf("  - Log-Vol RMSE on outlier ticks (Robust OCSN should help)\n");
    printf("  - ESS collapse count (Robust OCSN prevents particle death)\n");
    printf("  - mu_vol error convergence (Forgetting helps adaptation)\n");

    return 0;
}