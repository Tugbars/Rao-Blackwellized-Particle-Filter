/*=============================================================================
 * RBPF Comprehensive Comparison Test
 *
 * Compares 4 configurations:
 *   1. RBPF Baseline     - No parameter learning (fixed params)
 *   2. RBPF + Liu-West   - Original online learning
 *   3. RBPF + Storvik    - Sleeping Storvik learning
 *   4. RBPF + Storvik + APF - Full stack with lookahead
 *
 * Outputs CSV for analysis:
 *   - rbpf_baseline.csv
 *   - rbpf_liu_west.csv
 *   - rbpf_storvik.csv
 *   - rbpf_storvik_apf.csv
 *   - rbpf_comparison_summary.csv
 *
 * Scenarios tested (7000 ticks total):
 *   1. Calm period (R0)
 *   2. Gradual volatility increase (R0 → R1 → R2)
 *   3. Sudden crisis spike (R2 → R3)
 *   4. Crisis persistence
 *   5. Recovery (R3 → R2 → R1 → R0)
 *   6. Flash crash (brief R3 spike)
 *   7. Choppy regime switching
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * BUILD
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Requires: rbpf_ksc.c, rbpf_apf.c, rbpf_param_learn.c, rbpf_ksc_param_integration.c
 *
 * Windows (MSVC + Intel MKL):
 *   cmake --build . --config Release --target rbpf_comparison_test
 *
 * Linux (GCC + Intel MKL):
 *   source /opt/intel/oneapi/setvars.sh
 *   make rbpf_comparison_test
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * USAGE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   rbpf_comparison_test [seed] [output_dir]
 *
 *   Examples:
 *     rbpf_comparison_test                    # seed=42, output to current dir
 *     rbpf_comparison_test 123 ./results      # seed=123, output to ./results/
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
 * SYNTHETIC DATA GENERATION
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    rbpf_real_t *returns;
    rbpf_real_t *true_log_vol;
    rbpf_real_t *true_vol;
    int *true_regime;
    int n_ticks;
    int scenario_starts[10];
    int n_scenarios;
} SyntheticData;

/* PCG32 RNG */
typedef struct
{
    uint64_t state;
    uint64_t inc;
} pcg32_t;

static uint32_t pcg32_random(pcg32_t *rng)
{
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
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

/* Ground truth regime parameters */
static const rbpf_real_t TRUE_THETA[4] = {0.05f, 0.08f, 0.12f, 0.15f};
static const rbpf_real_t TRUE_MU_VOL[4] = {-4.6f, -3.5f, -2.5f, -1.5f};
static const rbpf_real_t TRUE_SIGMA_VOL[4] = {0.05f, 0.10f, 0.20f, 0.35f};

static SyntheticData *generate_test_data(int seed)
{
    SyntheticData *data = (SyntheticData *)calloc(1, sizeof(SyntheticData));

    int n = 7000;
    data->n_ticks = n;
    data->returns = (rbpf_real_t *)malloc(n * sizeof(rbpf_real_t));
    data->true_log_vol = (rbpf_real_t *)malloc(n * sizeof(rbpf_real_t));
    data->true_vol = (rbpf_real_t *)malloc(n * sizeof(rbpf_real_t));
    data->true_regime = (int *)malloc(n * sizeof(int));

    pcg32_t rng = {seed * 12345ULL + 1, seed * 67890ULL | 1};

    rbpf_real_t log_vol = TRUE_MU_VOL[0];
    int regime = 0;
    int t = 0;

    /* Scenario 1: Calm (0-999) */
    data->scenario_starts[0] = 0;
    data->n_scenarios = 1;

    for (; t < 1000; t++)
    {
        regime = 0;
        rbpf_real_t theta = TRUE_THETA[regime];
        rbpf_real_t mu = TRUE_MU_VOL[regime];
        rbpf_real_t sigma = TRUE_SIGMA_VOL[regime];

        log_vol = (1.0f - theta) * log_vol + theta * mu + sigma * (rbpf_real_t)pcg32_gaussian(&rng);
        rbpf_real_t vol = rbpf_exp(log_vol);
        rbpf_real_t ret = vol * (rbpf_real_t)pcg32_gaussian(&rng);

        data->returns[t] = ret;
        data->true_log_vol[t] = log_vol;
        data->true_vol[t] = vol;
        data->true_regime[t] = regime;
    }

    /* Scenario 2: Gradual increase (1000-1999) */
    data->scenario_starts[1] = 1000;
    data->n_scenarios = 2;

    for (; t < 2000; t++)
    {
        if (t < 1200)
            regime = 0;
        else if (t < 1600)
            regime = 1;
        else
            regime = 2;

        rbpf_real_t theta = TRUE_THETA[regime];
        rbpf_real_t mu = TRUE_MU_VOL[regime];
        rbpf_real_t sigma = TRUE_SIGMA_VOL[regime];

        log_vol = (1.0f - theta) * log_vol + theta * mu + sigma * (rbpf_real_t)pcg32_gaussian(&rng);
        rbpf_real_t vol = rbpf_exp(log_vol);
        rbpf_real_t ret = vol * (rbpf_real_t)pcg32_gaussian(&rng);

        data->returns[t] = ret;
        data->true_log_vol[t] = log_vol;
        data->true_vol[t] = vol;
        data->true_regime[t] = regime;
    }

    /* Scenario 3: Sudden crisis (2000-2499) */
    data->scenario_starts[2] = 2000;
    data->n_scenarios = 3;

    for (; t < 2500; t++)
    {
        regime = 3;
        rbpf_real_t theta = TRUE_THETA[regime];
        rbpf_real_t mu = TRUE_MU_VOL[regime];
        rbpf_real_t sigma = TRUE_SIGMA_VOL[regime];

        log_vol = (1.0f - theta) * log_vol + theta * mu + sigma * (rbpf_real_t)pcg32_gaussian(&rng);
        rbpf_real_t vol = rbpf_exp(log_vol);
        rbpf_real_t ret = vol * (rbpf_real_t)pcg32_gaussian(&rng);

        data->returns[t] = ret;
        data->true_log_vol[t] = log_vol;
        data->true_vol[t] = vol;
        data->true_regime[t] = regime;
    }

    /* Scenario 4: Crisis persistence (2500-3499) */
    data->scenario_starts[3] = 2500;
    data->n_scenarios = 4;

    for (; t < 3500; t++)
    {
        if (pcg32_double(&rng) < 0.1 && regime == 3)
            regime = 2;
        else if (pcg32_double(&rng) < 0.3 && regime == 2)
            regime = 3;
        else if (regime < 2)
            regime = 3;

        rbpf_real_t theta = TRUE_THETA[regime];
        rbpf_real_t mu = TRUE_MU_VOL[regime];
        rbpf_real_t sigma = TRUE_SIGMA_VOL[regime];

        log_vol = (1.0f - theta) * log_vol + theta * mu + sigma * (rbpf_real_t)pcg32_gaussian(&rng);
        rbpf_real_t vol = rbpf_exp(log_vol);
        rbpf_real_t ret = vol * (rbpf_real_t)pcg32_gaussian(&rng);

        data->returns[t] = ret;
        data->true_log_vol[t] = log_vol;
        data->true_vol[t] = vol;
        data->true_regime[t] = regime;
    }

    /* Scenario 5: Recovery (3500-4499) */
    data->scenario_starts[4] = 3500;
    data->n_scenarios = 5;

    for (; t < 4500; t++)
    {
        if (t < 3800)
            regime = 2;
        else if (t < 4200)
            regime = 1;
        else
            regime = 0;

        rbpf_real_t theta = TRUE_THETA[regime];
        rbpf_real_t mu = TRUE_MU_VOL[regime];
        rbpf_real_t sigma = TRUE_SIGMA_VOL[regime];

        log_vol = (1.0f - theta) * log_vol + theta * mu + sigma * (rbpf_real_t)pcg32_gaussian(&rng);
        rbpf_real_t vol = rbpf_exp(log_vol);
        rbpf_real_t ret = vol * (rbpf_real_t)pcg32_gaussian(&rng);

        data->returns[t] = ret;
        data->true_log_vol[t] = log_vol;
        data->true_vol[t] = vol;
        data->true_regime[t] = regime;
    }

    /* Scenario 6: Flash crash (4500-4999) */
    data->scenario_starts[5] = 4500;
    data->n_scenarios = 6;

    for (; t < 5000; t++)
    {
        if (t >= 4700 && t < 4750)
            regime = 3;
        else
            regime = 0;

        rbpf_real_t theta = TRUE_THETA[regime];
        rbpf_real_t mu = TRUE_MU_VOL[regime];
        rbpf_real_t sigma = TRUE_SIGMA_VOL[regime];

        log_vol = (1.0f - theta) * log_vol + theta * mu + sigma * (rbpf_real_t)pcg32_gaussian(&rng);
        rbpf_real_t vol = rbpf_exp(log_vol);
        rbpf_real_t ret = vol * (rbpf_real_t)pcg32_gaussian(&rng);

        data->returns[t] = ret;
        data->true_log_vol[t] = log_vol;
        data->true_vol[t] = vol;
        data->true_regime[t] = regime;
    }

    /* Scenario 7: Choppy switching (5000-6999) */
    data->scenario_starts[6] = 5000;
    data->n_scenarios = 7;

    int next_switch = 5000 + 50 + (int)(pcg32_double(&rng) * 100);
    regime = 1;

    for (; t < 7000; t++)
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

        rbpf_real_t theta = TRUE_THETA[regime];
        rbpf_real_t mu = TRUE_MU_VOL[regime];
        rbpf_real_t sigma = TRUE_SIGMA_VOL[regime];

        log_vol = (1.0f - theta) * log_vol + theta * mu + sigma * (rbpf_real_t)pcg32_gaussian(&rng);
        rbpf_real_t vol = rbpf_exp(log_vol);
        rbpf_real_t ret = vol * (rbpf_real_t)pcg32_gaussian(&rng);

        data->returns[t] = ret;
        data->true_log_vol[t] = log_vol;
        data->true_vol[t] = vol;
        data->true_regime[t] = regime;
    }

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

    /* Filter estimates */
    rbpf_real_t est_log_vol;
    rbpf_real_t est_vol;
    rbpf_real_t log_vol_var;
    rbpf_real_t ess;

    /* Regime */
    int est_regime;
    rbpf_real_t regime_prob_0;
    rbpf_real_t regime_prob_1;
    rbpf_real_t regime_prob_2;
    rbpf_real_t regime_prob_3;
    rbpf_real_t regime_entropy;

    /* Learned parameters (regime 0 as example) */
    rbpf_real_t learned_mu_vol_r0;
    rbpf_real_t learned_sigma_vol_r0;
    rbpf_real_t learned_mu_vol_r3;
    rbpf_real_t learned_sigma_vol_r3;

    /* Detection */
    rbpf_real_t surprise;
    rbpf_real_t vol_ratio;
    int change_detected;

    /* Timing */
    double latency_us;
    int resampled;

} TickRecord;

/*─────────────────────────────────────────────────────────────────────────────
 * TEST MODES
 *───────────────────────────────────────────────────────────────────────────*/

typedef enum
{
    MODE_BASELINE = 0,     /* RBPF alone, true params */
    MODE_BASELINE_MISSPEC, /* RBPF alone, wrong params (realistic) */
    MODE_LIU_WEST,         /* RBPF + Liu-West */
    MODE_STORVIK,          /* RBPF + Storvik */
    MODE_STORVIK_APF,      /* RBPF + Storvik + APF */
    NUM_MODES
} TestMode;

static const char *mode_names[] = {
    "Baseline",
    "Misspec",
    "Liu-West",
    "Storvik",
    "Storvik+APF"};

/*─────────────────────────────────────────────────────────────────────────────
 * RUN TEST
 *───────────────────────────────────────────────────────────────────────────*/

static void run_test(SyntheticData *data, TestMode mode, TickRecord *records,
                     double *total_time_us, double *max_latency_us)
{
    const int N_PARTICLES = 500;
    const int N_REGIMES = 4;

    RBPF_Extended *ext = NULL;
    RBPF_KSC *rbpf_raw = NULL; /* For baseline/Liu-West modes */

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

    case MODE_BASELINE_MISSPEC:
        /* Realistic scenario: parameters are 20-30% wrong
         * This demonstrates why online learning matters.
         *
         * Errors:
         *   mu_vol:    20% too high (less volatile than reality)
         *   sigma_vol: 30% too low (underestimate noise)
         *   theta:     same (hardest to estimate anyway)
         */
        rbpf_raw = rbpf_ksc_create(N_PARTICLES, N_REGIMES);
        for (int r = 0; r < N_REGIMES; r++)
        {
            rbpf_real_t wrong_mu = TRUE_MU_VOL[r] * 0.8f;       /* 20% too high (closer to 0) */
            rbpf_real_t wrong_sigma = TRUE_SIGMA_VOL[r] * 0.7f; /* 30% too low */
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
        rbpf_ksc_enable_liu_west(rbpf_raw, 0.98f, 100); /* shrinkage=0.98, warmup=100 */
        rbpf_ksc_init(rbpf_raw, TRUE_MU_VOL[0], 0.1f);
        break;

    case MODE_STORVIK:
    case MODE_STORVIK_APF:
        ext = rbpf_ext_create(N_PARTICLES, N_REGIMES, RBPF_PARAM_STORVIK);
        for (int r = 0; r < N_REGIMES; r++)
        {
            rbpf_ext_set_regime_params(ext, r, TRUE_THETA[r], TRUE_MU_VOL[r], TRUE_SIGMA_VOL[r]);
        }
        rbpf_ext_build_transition_lut(ext, trans);
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
        case MODE_BASELINE_MISSPEC:
        case MODE_LIU_WEST:
            rbpf_ksc_step(rbpf_raw, ret, &output);
            break;

        case MODE_STORVIK:
            rbpf_ext_step(ext, ret, &output);
            break;

        case MODE_STORVIK_APF:
            if (t < n - 1)
            {
                rbpf_ext_step_apf(ext, ret, data->returns[t + 1], &output);
            }
            else
            {
                rbpf_ext_step(ext, ret, &output);
            }
            break;

        default:
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

        /* Filter estimates */
        rec->est_log_vol = output.log_vol_mean;
        rec->est_vol = output.vol_mean;
        rec->log_vol_var = output.log_vol_var;
        rec->ess = output.ess;

        /* Regime */
        rec->est_regime = output.dominant_regime;
        rec->regime_prob_0 = output.regime_probs[0];
        rec->regime_prob_1 = output.regime_probs[1];
        rec->regime_prob_2 = output.regime_probs[2];
        rec->regime_prob_3 = output.regime_probs[3];
        rec->regime_entropy = output.regime_entropy;

        /* Learned params */
        rec->learned_mu_vol_r0 = output.learned_mu_vol[0];
        rec->learned_sigma_vol_r0 = output.learned_sigma_vol[0];
        rec->learned_mu_vol_r3 = output.learned_mu_vol[3];
        rec->learned_sigma_vol_r3 = output.learned_sigma_vol[3];

        /* Detection */
        rec->surprise = output.surprise;
        rec->vol_ratio = output.vol_ratio;
        rec->change_detected = output.regime_changed;

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
    fprintf(f, "tick,true_log_vol,true_vol,true_regime,return,"
               "est_log_vol,est_vol,log_vol_var,ess,"
               "est_regime,regime_prob_0,regime_prob_1,regime_prob_2,regime_prob_3,regime_entropy,"
               "learned_mu_vol_r0,learned_sigma_vol_r0,learned_mu_vol_r3,learned_sigma_vol_r3,"
               "surprise,vol_ratio,change_detected,latency_us,resampled\n");

    for (int t = 0; t < n; t++)
    {
        TickRecord *r = &records[t];
        fprintf(f, "%d,%.6f,%.6f,%d,%.8f,"
                   "%.6f,%.6f,%.6f,%.2f,"
                   "%d,%.4f,%.4f,%.4f,%.4f,%.4f,"
                   "%.4f,%.4f,%.4f,%.4f,"
                   "%.4f,%.4f,%d,%.2f,%d\n",
                r->tick, r->true_log_vol, r->true_vol, r->true_regime, r->return_val,
                r->est_log_vol, r->est_vol, r->log_vol_var, r->ess,
                r->est_regime, r->regime_prob_0, r->regime_prob_1, r->regime_prob_2, r->regime_prob_3, r->regime_entropy,
                r->learned_mu_vol_r0, r->learned_sigma_vol_r0, r->learned_mu_vol_r3, r->learned_sigma_vol_r3,
                r->surprise, r->vol_ratio, r->change_detected, r->latency_us, r->resampled);
    }

    fclose(f);
    printf("  Written: %s (%d rows)\n", filename, n);
}

/*─────────────────────────────────────────────────────────────────────────────
 * COMPUTE METRICS
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    double log_vol_rmse;
    double log_vol_mae;
    double vol_rmse;
    double vol_mae;
    double regime_accuracy;
    double avg_latency_us;
    double max_latency_us;
    double p99_latency_us;
    double avg_ess;
    double min_ess;
    int resample_count;

    /* Parameter learning metrics */
    double mu_vol_r0_error; /* Error vs true at end */
    double mu_vol_r3_error;

} SummaryMetrics;

static int compare_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static void compute_summary(TickRecord *records, SyntheticData *data, SummaryMetrics *m)
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
        if (r->resampled)
            m->resample_count++;

        latencies[t] = r->latency_us;
        sum_latency += r->latency_us;
        if (r->latency_us > max_latency)
            max_latency = r->latency_us;
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

    /* P99 latency */
    qsort(latencies, n, sizeof(double), compare_double);
    m->p99_latency_us = latencies[(int)(0.99 * n)];
    free(latencies);

    /* Parameter learning error at end of test */
    TickRecord *last = &records[n - 1];
    m->mu_vol_r0_error = fabs(last->learned_mu_vol_r0 - TRUE_MU_VOL[0]);
    m->mu_vol_r3_error = fabs(last->learned_mu_vol_r3 - TRUE_MU_VOL[3]);
}

static void write_summary_csv(const char *filename, SummaryMetrics *metrics, int n_modes)
{
    FILE *f = fopen(filename, "w");
    if (!f)
    {
        fprintf(stderr, "Failed to open %s\n", filename);
        return;
    }

    fprintf(f, "metric,baseline,liu_west,storvik,storvik_apf\n");

    fprintf(f, "log_vol_rmse,%.6f,%.6f,%.6f,%.6f\n",
            metrics[0].log_vol_rmse, metrics[1].log_vol_rmse,
            metrics[2].log_vol_rmse, metrics[3].log_vol_rmse);
    fprintf(f, "log_vol_mae,%.6f,%.6f,%.6f,%.6f\n",
            metrics[0].log_vol_mae, metrics[1].log_vol_mae,
            metrics[2].log_vol_mae, metrics[3].log_vol_mae);
    fprintf(f, "vol_rmse,%.6f,%.6f,%.6f,%.6f\n",
            metrics[0].vol_rmse, metrics[1].vol_rmse,
            metrics[2].vol_rmse, metrics[3].vol_rmse);
    fprintf(f, "regime_accuracy,%.4f,%.4f,%.4f,%.4f\n",
            metrics[0].regime_accuracy, metrics[1].regime_accuracy,
            metrics[2].regime_accuracy, metrics[3].regime_accuracy);
    fprintf(f, "avg_latency_us,%.2f,%.2f,%.2f,%.2f\n",
            metrics[0].avg_latency_us, metrics[1].avg_latency_us,
            metrics[2].avg_latency_us, metrics[3].avg_latency_us);
    fprintf(f, "p99_latency_us,%.2f,%.2f,%.2f,%.2f\n",
            metrics[0].p99_latency_us, metrics[1].p99_latency_us,
            metrics[2].p99_latency_us, metrics[3].p99_latency_us);
    fprintf(f, "avg_ess,%.1f,%.1f,%.1f,%.1f\n",
            metrics[0].avg_ess, metrics[1].avg_ess,
            metrics[2].avg_ess, metrics[3].avg_ess);
    fprintf(f, "min_ess,%.1f,%.1f,%.1f,%.1f\n",
            metrics[0].min_ess, metrics[1].min_ess,
            metrics[2].min_ess, metrics[3].min_ess);
    fprintf(f, "mu_vol_r0_error,%.4f,%.4f,%.4f,%.4f\n",
            metrics[0].mu_vol_r0_error, metrics[1].mu_vol_r0_error,
            metrics[2].mu_vol_r0_error, metrics[3].mu_vol_r0_error);
    fprintf(f, "mu_vol_r3_error,%.4f,%.4f,%.4f,%.4f\n",
            metrics[0].mu_vol_r3_error, metrics[1].mu_vol_r3_error,
            metrics[2].mu_vol_r3_error, metrics[3].mu_vol_r3_error);

    fclose(f);
    printf("  Written: %s\n", filename);
}

/*─────────────────────────────────────────────────────────────────────────────
 * MAIN
 *───────────────────────────────────────────────────────────────────────────*/

int main(int argc, char **argv)
{
    int seed = 42;
    const char *output_dir = "../../test/csv"; /* Default: root/test/csv from Build/Release */

    if (argc > 1)
        seed = atoi(argv[1]);
    if (argc > 2)
        output_dir = argv[2];

    init_timer();

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  RBPF Comprehensive Comparison Test\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Seed: %d\n", seed);
    printf("  Output: %s\n\n", output_dir);

    /* Generate data */
    printf("Generating synthetic data...\n");
    SyntheticData *data = generate_test_data(seed);
    printf("  Ticks: %d\n", data->n_ticks);
    printf("  Scenarios: %d\n\n", data->n_scenarios);

    int n = data->n_ticks;
    TickRecord *records[NUM_MODES];
    SummaryMetrics metrics[NUM_MODES];
    double total_time[NUM_MODES], max_latency[NUM_MODES];

    const char *csv_names[] = {
        "rbpf_baseline.csv",
        "rbpf_misspec.csv",
        "rbpf_liu_west.csv",
        "rbpf_storvik.csv",
        "rbpf_storvik_apf.csv"};

    /* Run all modes */
    for (int mode = 0; mode < NUM_MODES; mode++)
    {
        printf("Running %s...\n", mode_names[mode]);
        records[mode] = (TickRecord *)calloc(n, sizeof(TickRecord));

        run_test(data, (TestMode)mode, records[mode], &total_time[mode], &max_latency[mode]);
        compute_summary(records[mode], data, &metrics[mode]);

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
    write_summary_csv(summary_path, metrics, NUM_MODES);

    /* Print comparison table */
    printf("\n════════════════════════════════════════════════════════════════════════════\n");
    printf("  SUMMARY COMPARISON\n");
    printf("════════════════════════════════════════════════════════════════════════════\n");
    printf("%-20s %10s %10s %10s %10s %12s\n", "Metric", "Baseline", "Misspec", "Liu-West", "Storvik", "Storvik+APF");
    printf("────────────────────────────────────────────────────────────────────────────\n");
    printf("%-20s %10.4f %10.4f %10.4f %10.4f %12.4f\n", "Log-Vol RMSE",
           metrics[0].log_vol_rmse, metrics[1].log_vol_rmse,
           metrics[2].log_vol_rmse, metrics[3].log_vol_rmse, metrics[4].log_vol_rmse);
    printf("%-20s %10.4f %10.4f %10.4f %10.4f %12.4f\n", "Vol RMSE",
           metrics[0].vol_rmse, metrics[1].vol_rmse,
           metrics[2].vol_rmse, metrics[3].vol_rmse, metrics[4].vol_rmse);
    printf("%-20s %9.1f%% %9.1f%% %9.1f%% %9.1f%% %11.1f%%\n", "Regime Accuracy",
           100 * metrics[0].regime_accuracy, 100 * metrics[1].regime_accuracy,
           100 * metrics[2].regime_accuracy, 100 * metrics[3].regime_accuracy, 100 * metrics[4].regime_accuracy);
    printf("%-20s %10.2f %10.2f %10.2f %10.2f %12.2f\n", "Avg Latency (us)",
           metrics[0].avg_latency_us, metrics[1].avg_latency_us,
           metrics[2].avg_latency_us, metrics[3].avg_latency_us, metrics[4].avg_latency_us);
    printf("%-20s %10.2f %10.2f %10.2f %10.2f %12.2f\n", "P99 Latency (us)",
           metrics[0].p99_latency_us, metrics[1].p99_latency_us,
           metrics[2].p99_latency_us, metrics[3].p99_latency_us, metrics[4].p99_latency_us);
    printf("%-20s %10.1f %10.1f %10.1f %10.1f %12.1f\n", "Avg ESS",
           metrics[0].avg_ess, metrics[1].avg_ess,
           metrics[2].avg_ess, metrics[3].avg_ess, metrics[4].avg_ess);
    printf("%-20s %10.4f %10.4f %10.4f %10.4f %12.4f\n", "mu_vol R0 Error",
           metrics[0].mu_vol_r0_error, metrics[1].mu_vol_r0_error,
           metrics[2].mu_vol_r0_error, metrics[3].mu_vol_r0_error, metrics[4].mu_vol_r0_error);
    printf("%-20s %10.4f %10.4f %10.4f %10.4f %12.4f\n", "mu_vol R3 Error",
           metrics[0].mu_vol_r3_error, metrics[1].mu_vol_r3_error,
           metrics[2].mu_vol_r3_error, metrics[3].mu_vol_r3_error, metrics[4].mu_vol_r3_error);
    printf("════════════════════════════════════════════════════════════════════════════\n");

    /* Scenario breakdown */
    printf("\nSCENARIO BREAKDOWN (Regime Accuracy %%)\n");
    printf("────────────────────────────────────────────────────────────────────────────\n");

    const char *scenario_names[] = {
        "1. Calm",
        "2. Gradual Rise",
        "3. Sudden Crisis",
        "4. Crisis Persist",
        "5. Recovery",
        "6. Flash Crash",
        "7. Choppy"};

    printf("%-20s %10s %10s %10s %10s %12s\n", "Scenario", "Baseline", "Misspec", "Liu-West", "Storvik", "Storvik+APF");
    printf("────────────────────────────────────────────────────────────────────────────\n");

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

        printf("%-20s %9.1f%% %9.1f%% %9.1f%% %9.1f%% %11.1f%%\n", scenario_names[s],
               100.0 * correct[0] / count,
               100.0 * correct[1] / count,
               100.0 * correct[2] / count,
               100.0 * correct[3] / count,
               100.0 * correct[4] / count);
    }
    printf("════════════════════════════════════════════════════════════════════════════\n");

    /* Cleanup */
    for (int m = 0; m < NUM_MODES; m++)
        free(records[m]);
    free_synthetic_data(data);

    printf("\nDone. Open CSV files for detailed analysis.\n");

    return 0;
}