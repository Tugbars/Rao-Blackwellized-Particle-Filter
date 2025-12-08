/*=============================================================================
 * RBPF Pipeline Comprehensive Test
 *
 * Outputs CSV for Jupyter notebook analysis:
 *   - rbpf_standard.csv  : Pipeline with standard RBPF
 *   - rbpf_apf.csv       : Pipeline with APF extension
 *   - rbpf_summary.csv   : Aggregate metrics comparison
 *
 * Scenarios tested:
 *   1. Calm period (R0)
 *   2. Gradual volatility increase (R0 → R1 → R2)
 *   3. Sudden crisis spike (R2 → R3)
 *   4. Crisis persistence
 *   5. Recovery (R3 → R2 → R1 → R0)
 *   6. Flash crash (brief R3 spike)
 *   7. Choppy regime switching
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * BUILD INSTRUCTIONS
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Project structure:
 *   project_root/
 *   ├── CMakeLists.txt          (root cmake)
 *   ├── RBPF/
 *   │   ├── CMakeLists.txt      (library cmake)
 *   │   ├── rbpf_ksc.h
 *   │   ├── rbpf_ksc.c
 *   │   ├── rbpf_apf.c
 *   │   ├── rbpf_pipeline.c
 *   │   └── mkl_config.h
 *   └── test/
 *       ├── rbpf_pipeline_test.c   <-- this file
 *       ├── rbpf_analysis.ipynb
 *       └── ...
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Windows (Visual Studio + Intel oneAPI MKL):
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   # From project root:
 *   mkdir build
 *   cd build
 *   cmake .. -G "Visual Studio 17 2022" -A x64
 *   cmake --build . --config Release
 *
 *   # Run:
 *   Release\rbpf_pipeline_test.exe
 *
 *   # Or with Ninja (faster):
 *   cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
 *   ninja
 *   ./rbpf_pipeline_test.exe
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Linux (GCC + Intel oneAPI MKL):
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   # Setup MKL environment:
 *   source /opt/intel/oneapi/setvars.sh
 *
 *   # Build:
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   make -j$(nproc)
 *
 *   # Run:
 *   ./rbpf_pipeline_test
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Manual build (without CMake):
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   # Linux/GCC:
 *   cd test
 *   gcc -O3 -march=native -fopenmp -I../RBPF \
 *       rbpf_pipeline_test.c \
 *       ../RBPF/rbpf_ksc.c \
 *       ../RBPF/rbpf_apf.c \
 *       ../RBPF/rbpf_pipeline.c \
 *       -o rbpf_pipeline_test \
 *       -lmkl_rt -lpthread -lm -ldl
 *
 *   # Windows/MSVC (Developer Command Prompt):
 *   cd test
 *   cl /O2 /openmp /I..\RBPF ^
 *       rbpf_pipeline_test.c ^
 *       ..\RBPF\rbpf_ksc.c ^
 *       ..\RBPF\rbpf_apf.c ^
 *       ..\RBPF\rbpf_pipeline.c ^
 *       /Fe:rbpf_pipeline_test.exe ^
 *       mkl_rt.lib
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * OUTPUT FILES
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   rbpf_standard.csv  - 7000 rows × 29 columns (standard RBPF)
 *   rbpf_apf.csv       - 7000 rows × 29 columns (APF mode)
 *   rbpf_summary.csv   - Aggregate metrics comparison
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * USAGE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   rbpf_pipeline_test [seed] [output_dir]
 *
 *   Arguments:
 *     seed       - Random seed (default: 42)
 *     output_dir - Directory for CSV output (default: current directory)
 *
 *   Examples:
 *     rbpf_pipeline_test                    # seed=42, output to current dir
 *     rbpf_pipeline_test 123                # seed=123, output to current dir
 *     rbpf_pipeline_test 42 ../../test      # seed=42, output to ../../test/
 *     rbpf_pipeline_test 42 C:\project\test # absolute path on Windows
 *
 * Open rbpf_analysis.ipynb in Jupyter to visualize results.
 *
 * Author: RBPF-KSC Project
 *===========================================================================*/

#include "rbpf_ksc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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
    g_timer_freq = (double)freq.QuadPart / 1e6; /* Convert to μs */
}

static inline double get_time_us(void)
{
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / g_timer_freq;
}
#else
#include <sys/time.h>

static void init_timer(void) { /* No-op on Unix */ }

static inline double get_time_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}
#endif

/*─────────────────────────────────────────────────────────────────────────────
 * SYNTHETIC DATA GENERATION
 *
 * Generates returns from known stochastic volatility process with regime changes.
 * This gives us ground truth for evaluation.
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    rbpf_real_t *returns;      /* Observed returns */
    rbpf_real_t *true_log_vol; /* True latent log-volatility */
    rbpf_real_t *true_vol;     /* True volatility (exp of above) */
    int *true_regime;          /* True regime at each tick */
    int n_ticks;

    /* Scenario boundaries for analysis */
    int scenario_starts[10];
    int n_scenarios;
} SyntheticData;

/* PCG32 for reproducible data generation */
typedef struct
{
    uint64_t state;
    uint64_t inc;
} pcg32_random_t;

static uint32_t pcg32_random_r(pcg32_random_t *rng)
{
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static double pcg32_double(pcg32_random_t *rng)
{
    return (double)pcg32_random_r(rng) / 4294967296.0;
}

static double pcg32_gaussian(pcg32_random_t *rng)
{
    double u1 = pcg32_double(rng);
    double u2 = pcg32_double(rng);
    if (u1 < 1e-10)
        u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
}

/* Regime parameters for data generation (ground truth) */
static const rbpf_real_t TRUE_THETA[4] = {0.05f, 0.08f, 0.12f, 0.15f};
static const rbpf_real_t TRUE_MU_VOL[4] = {-4.6f, -3.5f, -2.5f, -1.5f};
static const rbpf_real_t TRUE_SIGMA_VOL[4] = {0.05f, 0.10f, 0.20f, 0.35f};

static SyntheticData *generate_test_data(int seed)
{
    SyntheticData *data = (SyntheticData *)calloc(1, sizeof(SyntheticData));

    /* Total ticks: 7000 (covers all scenarios with margin) */
    int n = 7000;
    data->n_ticks = n;
    data->returns = (rbpf_real_t *)malloc(n * sizeof(rbpf_real_t));
    data->true_log_vol = (rbpf_real_t *)malloc(n * sizeof(rbpf_real_t));
    data->true_vol = (rbpf_real_t *)malloc(n * sizeof(rbpf_real_t));
    data->true_regime = (int *)malloc(n * sizeof(int));

    pcg32_random_t rng = {seed * 12345ULL + 1, seed * 67890ULL | 1};

    /* Initialize */
    rbpf_real_t log_vol = TRUE_MU_VOL[0];
    int regime = 0;
    int t = 0;

    /*========================================================================
     * SCENARIO 1: Calm period (ticks 0-999)
     * Pure R0, establishes baseline
     *======================================================================*/
    data->scenario_starts[0] = 0;
    data->n_scenarios = 1;

    for (; t < 1000; t++)
    {
        regime = 0;
        rbpf_real_t theta = TRUE_THETA[regime];
        rbpf_real_t mu = TRUE_MU_VOL[regime];
        rbpf_real_t sigma = TRUE_SIGMA_VOL[regime];

        /* OU dynamics */
        log_vol = (1.0f - theta) * log_vol + theta * mu + sigma * pcg32_gaussian(&rng);
        rbpf_real_t vol = expf(log_vol);

        /* Generate return */
        rbpf_real_t ret = vol * pcg32_gaussian(&rng);

        data->returns[t] = ret;
        data->true_log_vol[t] = log_vol;
        data->true_vol[t] = vol;
        data->true_regime[t] = regime;
    }

    /*========================================================================
     * SCENARIO 2: Gradual increase (ticks 1000-1999)
     * R0 → R1 @ 1200, R1 → R2 @ 1600
     *======================================================================*/
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

        log_vol = (1.0f - theta) * log_vol + theta * mu + sigma * pcg32_gaussian(&rng);
        rbpf_real_t vol = expf(log_vol);
        rbpf_real_t ret = vol * pcg32_gaussian(&rng);

        data->returns[t] = ret;
        data->true_log_vol[t] = log_vol;
        data->true_vol[t] = vol;
        data->true_regime[t] = regime;
    }

    /*========================================================================
     * SCENARIO 3: Sudden crisis (ticks 2000-2499)
     * R2 → R3 instant @ 2000 (simulates flash event)
     *======================================================================*/
    data->scenario_starts[2] = 2000;
    data->n_scenarios = 3;

    for (; t < 2500; t++)
    {
        regime = 3;
        rbpf_real_t theta = TRUE_THETA[regime];
        rbpf_real_t mu = TRUE_MU_VOL[regime];
        rbpf_real_t sigma = TRUE_SIGMA_VOL[regime];

        log_vol = (1.0f - theta) * log_vol + theta * mu + sigma * pcg32_gaussian(&rng);
        rbpf_real_t vol = expf(log_vol);
        rbpf_real_t ret = vol * pcg32_gaussian(&rng);

        data->returns[t] = ret;
        data->true_log_vol[t] = log_vol;
        data->true_vol[t] = vol;
        data->true_regime[t] = regime;
    }

    /*========================================================================
     * SCENARIO 4: Crisis persistence (ticks 2500-3499)
     * Stay in R3 with occasional R2 dips
     *======================================================================*/
    data->scenario_starts[3] = 2500;
    data->n_scenarios = 4;

    for (; t < 3500; t++)
    {
        /* 90% R3, 10% R2 */
        if (pcg32_double(&rng) < 0.1 && regime == 3)
            regime = 2;
        else if (pcg32_double(&rng) < 0.3 && regime == 2)
            regime = 3;
        else if (regime < 2)
            regime = 3;

        rbpf_real_t theta = TRUE_THETA[regime];
        rbpf_real_t mu = TRUE_MU_VOL[regime];
        rbpf_real_t sigma = TRUE_SIGMA_VOL[regime];

        log_vol = (1.0f - theta) * log_vol + theta * mu + sigma * pcg32_gaussian(&rng);
        rbpf_real_t vol = expf(log_vol);
        rbpf_real_t ret = vol * pcg32_gaussian(&rng);

        data->returns[t] = ret;
        data->true_log_vol[t] = log_vol;
        data->true_vol[t] = vol;
        data->true_regime[t] = regime;
    }

    /*========================================================================
     * SCENARIO 5: Recovery (ticks 3500-4499)
     * R3 → R2 @ 3500, R2 → R1 @ 3800, R1 → R0 @ 4200
     *======================================================================*/
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

        log_vol = (1.0f - theta) * log_vol + theta * mu + sigma * pcg32_gaussian(&rng);
        rbpf_real_t vol = expf(log_vol);
        rbpf_real_t ret = vol * pcg32_gaussian(&rng);

        data->returns[t] = ret;
        data->true_log_vol[t] = log_vol;
        data->true_vol[t] = vol;
        data->true_regime[t] = regime;
    }

    /*========================================================================
     * SCENARIO 6: Flash crash (ticks 4500-4999)
     * Calm R0, then brief R3 spike @ 4700-4750, back to R0
     *======================================================================*/
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

        log_vol = (1.0f - theta) * log_vol + theta * mu + sigma * pcg32_gaussian(&rng);
        rbpf_real_t vol = expf(log_vol);
        rbpf_real_t ret = vol * pcg32_gaussian(&rng);

        data->returns[t] = ret;
        data->true_log_vol[t] = log_vol;
        data->true_vol[t] = vol;
        data->true_regime[t] = regime;
    }

    /*========================================================================
     * SCENARIO 7: Choppy switching (ticks 5000-6999)
     * Random regime changes every ~100 ticks (tests tracking agility)
     *======================================================================*/
    data->scenario_starts[6] = 5000;
    data->n_scenarios = 7;

    int next_switch = 5000 + 50 + (int)(pcg32_double(&rng) * 100);
    regime = 1;

    for (; t < 7000; t++)
    {
        if (t >= next_switch)
        {
            /* Random jump to adjacent regime */
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

        log_vol = (1.0f - theta) * log_vol + theta * mu + sigma * pcg32_gaussian(&rng);
        rbpf_real_t vol = expf(log_vol);
        rbpf_real_t ret = vol * pcg32_gaussian(&rng);

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
 * CSV OUTPUT STRUCTURES
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    int tick;

    /* Ground truth */
    rbpf_real_t true_log_vol;
    rbpf_real_t true_vol;
    int true_regime;
    rbpf_real_t return_val;

    /* Filter outputs */
    rbpf_real_t est_log_vol;
    rbpf_real_t est_log_vol_smooth;
    rbpf_real_t est_vol;
    rbpf_real_t est_vol_smooth;
    rbpf_real_t log_vol_var;
    rbpf_real_t log_vol_var_smooth;

    /* Regime */
    int est_regime_fast;
    int est_regime_smooth;
    rbpf_real_t regime_prob_0;
    rbpf_real_t regime_prob_1;
    rbpf_real_t regime_prob_2;
    rbpf_real_t regime_prob_3;
    rbpf_real_t regime_confidence;
    rbpf_real_t regime_entropy;

    /* Detection signals */
    rbpf_real_t surprise;
    rbpf_real_t vol_ratio;
    int change_detected;
    int change_type;

    /* Filter health */
    rbpf_real_t ess;
    int resampled;
    int apf_triggered;

    /* Position scaling */
    rbpf_real_t position_scale;
    int action;

    /* Timing */
    double latency_us;

} TickRecord;

/*─────────────────────────────────────────────────────────────────────────────
 * RUN TEST AND COLLECT DATA
 *───────────────────────────────────────────────────────────────────────────*/

static void run_pipeline_test(
    SyntheticData *data,
    int use_apf,
    TickRecord *records,
    double *total_time_us,
    double *max_latency_us)
{
    /* Create pipeline with default config */
    RBPF_PipelineConfig cfg = rbpf_pipeline_default_config();
    cfg.n_particles = 200;
    cfg.n_regimes = 4;
    cfg.smooth_lag = 5;
    cfg.enable_learning = 0; /* Test without Liu-West for fair comparison */

    RBPF_Pipeline *pipe = rbpf_pipeline_create(&cfg);
    if (!pipe)
    {
        fprintf(stderr, "Failed to create pipeline\n");
        return;
    }

    /* Initialize with true initial vol */
    rbpf_pipeline_init(pipe, data->true_vol[0]);

    *total_time_us = 0.0;
    *max_latency_us = 0.0;

    int n = data->n_ticks;

    for (int t = 0; t < n; t++)
    {
        RBPF_Signal sig;
        double t_start = get_time_us();

        if (use_apf && t < n - 1)
        {
            /* APF mode: use next return for lookahead */
            rbpf_pipeline_step_apf(pipe, data->returns[t], data->returns[t + 1], &sig);
        }
        else
        {
            /* Standard mode */
            rbpf_pipeline_step(pipe, data->returns[t], &sig);
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
        rec->return_val = data->returns[t];

        /* Filter estimates */
        rec->est_log_vol = sig.log_vol;
        rec->est_vol = sig.vol_forecast;
        rec->log_vol_var = sig.vol_uncertainty * sig.vol_uncertainty;

        /* Get smooth estimates from internal RBPF output */
        /* (Pipeline exposes these through the signal) */
        rec->est_log_vol_smooth = sig.log_vol; /* Pipeline uses smooth internally */
        rec->est_vol_smooth = sig.vol_forecast;
        rec->log_vol_var_smooth = rec->log_vol_var;

        /* Regime */
        rec->est_regime_fast = sig.regime;
        rec->est_regime_smooth = sig.regime;
        rec->regime_prob_0 = sig.regime_probs[0];
        rec->regime_prob_1 = sig.regime_probs[1];
        rec->regime_prob_2 = sig.regime_probs[2];
        rec->regime_prob_3 = sig.regime_probs[3];
        rec->regime_confidence = sig.regime_confidence;
        rec->regime_entropy = sig.regime_entropy;

        /* Detection */
        rec->surprise = sig.surprise;
        rec->vol_ratio = sig.vol_ratio;
        rec->change_detected = sig.change_detected;
        rec->change_type = sig.change_type;

        /* Health */
        rec->ess = sig.ess;
        rec->resampled = 0; /* Not directly exposed, could add */
        rec->apf_triggered = use_apf ? 1 : 0;

        /* Scaling */
        rec->position_scale = sig.position_scale;
        rec->action = sig.action;

        /* Timing */
        rec->latency_us = latency;
    }

    rbpf_pipeline_destroy(pipe);
}

/*─────────────────────────────────────────────────────────────────────────────
 * WRITE CSV FILES
 *───────────────────────────────────────────────────────────────────────────*/

static void write_tick_csv(const char *filename, TickRecord *records, int n)
{
    FILE *f = fopen(filename, "w");
    if (!f)
    {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }

    /* Header */
    fprintf(f, "tick,true_log_vol,true_vol,true_regime,return,"
               "est_log_vol,est_log_vol_smooth,est_vol,est_vol_smooth,"
               "log_vol_var,log_vol_var_smooth,"
               "est_regime_fast,est_regime_smooth,"
               "regime_prob_0,regime_prob_1,regime_prob_2,regime_prob_3,"
               "regime_confidence,regime_entropy,"
               "surprise,vol_ratio,change_detected,change_type,"
               "ess,resampled,apf_triggered,"
               "position_scale,action,latency_us\n");

    /* Data rows */
    for (int t = 0; t < n; t++)
    {
        TickRecord *r = &records[t];
        fprintf(f, "%d,%.6f,%.6f,%d,%.8f,"
                   "%.6f,%.6f,%.6f,%.6f,"
                   "%.6f,%.6f,"
                   "%d,%d,"
                   "%.4f,%.4f,%.4f,%.4f,"
                   "%.4f,%.4f,"
                   "%.4f,%.4f,%d,%d,"
                   "%.2f,%d,%d,"
                   "%.4f,%d,%.2f\n",
                r->tick, r->true_log_vol, r->true_vol, r->true_regime, r->return_val,
                r->est_log_vol, r->est_log_vol_smooth, r->est_vol, r->est_vol_smooth,
                r->log_vol_var, r->log_vol_var_smooth,
                r->est_regime_fast, r->est_regime_smooth,
                r->regime_prob_0, r->regime_prob_1, r->regime_prob_2, r->regime_prob_3,
                r->regime_confidence, r->regime_entropy,
                r->surprise, r->vol_ratio, r->change_detected, r->change_type,
                r->ess, r->resampled, r->apf_triggered,
                r->position_scale, r->action, r->latency_us);
    }

    fclose(f);
    printf("  Written: %s (%d rows)\n", filename, n);
}

/*─────────────────────────────────────────────────────────────────────────────
 * COMPUTE SUMMARY METRICS
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    /* Volatility tracking */
    double log_vol_rmse;
    double log_vol_mae;
    double log_vol_bias;
    double vol_rmse;
    double vol_mae;

    /* Regime detection */
    double regime_accuracy;
    double regime_detection_delay; /* Avg ticks to detect true change */
    int false_positives;
    int false_negatives;
    int true_positives;

    /* Timing */
    double avg_latency_us;
    double max_latency_us;
    double p99_latency_us;

    /* Filter health */
    double avg_ess;
    double min_ess;
    int resample_count;

} SummaryMetrics;

static void compute_summary(TickRecord *records, SyntheticData *data,
                            SummaryMetrics *metrics)
{
    int n = data->n_ticks;
    memset(metrics, 0, sizeof(SummaryMetrics));

    double sum_log_err = 0.0, sum_log_err2 = 0.0;
    double sum_vol_err = 0.0, sum_vol_err2 = 0.0;
    double sum_log_bias = 0.0;
    int regime_correct = 0;
    double sum_ess = 0.0;
    double min_ess = 1e9;
    double sum_latency = 0.0;
    double max_latency = 0.0;

    /* For percentile */
    double *latencies = (double *)malloc(n * sizeof(double));

    /* Track regime transitions for detection delay */
    int last_true_regime = data->true_regime[0];
    int last_est_regime = records[0].est_regime_smooth;
    int pending_detection = 0;
    int detection_start_tick = 0;
    double total_delay = 0.0;
    int n_transitions = 0;

    for (int t = 0; t < n; t++)
    {
        TickRecord *r = &records[t];

        /* Volatility errors */
        double log_err = r->est_log_vol - r->true_log_vol;
        double vol_err = r->est_vol - r->true_vol;

        sum_log_err += fabs(log_err);
        sum_log_err2 += log_err * log_err;
        sum_log_bias += log_err;
        sum_vol_err += fabs(vol_err);
        sum_vol_err2 += vol_err * vol_err;

        /* Regime accuracy */
        if (r->est_regime_smooth == r->true_regime)
            regime_correct++;

        /* Detection delay tracking */
        if (r->true_regime != last_true_regime)
        {
            /* True regime changed - start timing */
            pending_detection = 1;
            detection_start_tick = t;
            last_true_regime = r->true_regime;
            n_transitions++;
        }

        if (pending_detection && r->est_regime_smooth == r->true_regime)
        {
            /* Detected the change */
            total_delay += (t - detection_start_tick);
            pending_detection = 0;
        }

        /* ESS */
        sum_ess += r->ess;
        if (r->ess < min_ess)
            min_ess = r->ess;
        if (r->resampled)
            metrics->resample_count++;

        /* Latency */
        latencies[t] = r->latency_us;
        sum_latency += r->latency_us;
        if (r->latency_us > max_latency)
            max_latency = r->latency_us;

        /* False positive/negative for change detection */
        int true_change = (t > 0 && data->true_regime[t] != data->true_regime[t - 1]);
        int detected_change = (r->change_detected > 0);

        if (true_change && detected_change)
            metrics->true_positives++;
        else if (!true_change && detected_change)
            metrics->false_positives++;
        else if (true_change && !detected_change)
            metrics->false_negatives++;
    }

    /* Compute final metrics */
    metrics->log_vol_rmse = sqrt(sum_log_err2 / n);
    metrics->log_vol_mae = sum_log_err / n;
    metrics->log_vol_bias = sum_log_bias / n;
    metrics->vol_rmse = sqrt(sum_vol_err2 / n);
    metrics->vol_mae = sum_vol_err / n;

    metrics->regime_accuracy = (double)regime_correct / n;
    metrics->regime_detection_delay = (n_transitions > 0) ? total_delay / n_transitions : 0.0;

    metrics->avg_latency_us = sum_latency / n;
    metrics->max_latency_us = max_latency;

    /* P99 latency */
    /* Simple bubble sort for small n - replace with quickselect for large n */
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            if (latencies[j] < latencies[i])
            {
                double tmp = latencies[i];
                latencies[i] = latencies[j];
                latencies[j] = tmp;
            }
        }
    }
    int p99_idx = (int)(0.99 * n);
    metrics->p99_latency_us = latencies[p99_idx];

    metrics->avg_ess = sum_ess / n;
    metrics->min_ess = min_ess;

    free(latencies);
}

static void write_summary_csv(const char *filename,
                              SummaryMetrics *standard,
                              SummaryMetrics *apf)
{
    FILE *f = fopen(filename, "w");
    if (!f)
    {
        fprintf(stderr, "Failed to open %s\n", filename);
        return;
    }

    fprintf(f, "metric,standard,apf,improvement_pct\n");

/* Helper macro */
#define WRITE_METRIC(name, std_val, apf_val, lower_better)                                                                               \
    do                                                                                                                                   \
    {                                                                                                                                    \
        double imp = (lower_better) ? 100.0 * (std_val - apf_val) / (std_val + 1e-10) : 100.0 * (apf_val - std_val) / (std_val + 1e-10); \
        fprintf(f, "%s,%.6f,%.6f,%.2f\n", name, std_val, apf_val, imp);                                                                  \
    } while (0)

    WRITE_METRIC("log_vol_rmse", standard->log_vol_rmse, apf->log_vol_rmse, 1);
    WRITE_METRIC("log_vol_mae", standard->log_vol_mae, apf->log_vol_mae, 1);
    WRITE_METRIC("log_vol_bias", fabs(standard->log_vol_bias), fabs(apf->log_vol_bias), 1);
    WRITE_METRIC("vol_rmse", standard->vol_rmse, apf->vol_rmse, 1);
    WRITE_METRIC("vol_mae", standard->vol_mae, apf->vol_mae, 1);
    WRITE_METRIC("regime_accuracy", standard->regime_accuracy, apf->regime_accuracy, 0);
    WRITE_METRIC("detection_delay", standard->regime_detection_delay, apf->regime_detection_delay, 1);
    WRITE_METRIC("false_positives", (double)standard->false_positives, (double)apf->false_positives, 1);
    WRITE_METRIC("false_negatives", (double)standard->false_negatives, (double)apf->false_negatives, 1);
    WRITE_METRIC("true_positives", (double)standard->true_positives, (double)apf->true_positives, 0);
    WRITE_METRIC("avg_latency_us", standard->avg_latency_us, apf->avg_latency_us, 1);
    WRITE_METRIC("max_latency_us", standard->max_latency_us, apf->max_latency_us, 1);
    WRITE_METRIC("p99_latency_us", standard->p99_latency_us, apf->p99_latency_us, 1);
    WRITE_METRIC("avg_ess", standard->avg_ess, apf->avg_ess, 0);
    WRITE_METRIC("min_ess", standard->min_ess, apf->min_ess, 0);

#undef WRITE_METRIC

    fclose(f);
    printf("  Written: %s\n", filename);
}

/*─────────────────────────────────────────────────────────────────────────────
 * MAIN
 *───────────────────────────────────────────────────────────────────────────*/

int main(int argc, char **argv)
{
    int seed = 42;
    const char *output_dir = "."; /* Default: current directory */

    if (argc > 1)
        seed = atoi(argv[1]);
    if (argc > 2)
        output_dir = argv[2]; /* Optional output directory */

    /* Initialize high-resolution timer (needed for Windows) */
    init_timer();

    printf("=================================================================\n");
    printf("RBPF Pipeline Comprehensive Test\n");
    printf("=================================================================\n");
    printf("Seed: %d\n", seed);
    printf("Output: %s\n\n", output_dir);

    /* Generate test data */
    printf("Generating synthetic data...\n");
    SyntheticData *data = generate_test_data(seed);
    printf("  Ticks: %d\n", data->n_ticks);
    printf("  Scenarios: %d\n\n", data->n_scenarios);

    /* Allocate record buffers */
    int n = data->n_ticks;
    TickRecord *records_standard = (TickRecord *)calloc(n, sizeof(TickRecord));
    TickRecord *records_apf = (TickRecord *)calloc(n, sizeof(TickRecord));

    /* Run standard pipeline */
    printf("Running STANDARD pipeline...\n");
    double total_std, max_std;
    run_pipeline_test(data, 0, records_standard, &total_std, &max_std);
    printf("  Total time: %.2f ms\n", total_std / 1000.0);
    printf("  Avg latency: %.2f μs\n", total_std / n);
    printf("  Max latency: %.2f μs\n\n", max_std);

    /* Run APF pipeline */
    printf("Running APF pipeline...\n");
    double total_apf, max_apf;
    run_pipeline_test(data, 1, records_apf, &total_apf, &max_apf);
    printf("  Total time: %.2f ms\n", total_apf / 1000.0);
    printf("  Avg latency: %.2f μs\n", total_apf / n);
    printf("  Max latency: %.2f μs\n\n", max_apf);

    /* Compute summary metrics */
    printf("Computing summary metrics...\n");
    SummaryMetrics summary_std, summary_apf;
    compute_summary(records_standard, data, &summary_std);
    compute_summary(records_apf, data, &summary_apf);

    /* Write CSV files */
    printf("\nWriting CSV files to: %s\n", output_dir);

    char path_std[512], path_apf[512], path_summary[512];
    snprintf(path_std, sizeof(path_std), "%s/rbpf_standard.csv", output_dir);
    snprintf(path_apf, sizeof(path_apf), "%s/rbpf_apf.csv", output_dir);
    snprintf(path_summary, sizeof(path_summary), "%s/rbpf_summary.csv", output_dir);

    write_tick_csv(path_std, records_standard, n);
    write_tick_csv(path_apf, records_apf, n);
    write_summary_csv(path_summary, &summary_std, &summary_apf);

    /* Print summary comparison */
    printf("\n=================================================================\n");
    printf("SUMMARY COMPARISON\n");
    printf("=================================================================\n");
    printf("%-25s %12s %12s %12s\n", "Metric", "Standard", "APF", "Δ%%");
    printf("-----------------------------------------------------------------\n");
    printf("%-25s %12.4f %12.4f %+11.1f%%\n", "Log-Vol RMSE",
           summary_std.log_vol_rmse, summary_apf.log_vol_rmse,
           100.0 * (summary_std.log_vol_rmse - summary_apf.log_vol_rmse) / summary_std.log_vol_rmse);
    printf("%-25s %12.4f %12.4f %+11.1f%%\n", "Vol RMSE",
           summary_std.vol_rmse, summary_apf.vol_rmse,
           100.0 * (summary_std.vol_rmse - summary_apf.vol_rmse) / summary_std.vol_rmse);
    printf("%-25s %12.1f%% %11.1f%% %+11.1f%%\n", "Regime Accuracy",
           100.0 * summary_std.regime_accuracy, 100.0 * summary_apf.regime_accuracy,
           100.0 * (summary_apf.regime_accuracy - summary_std.regime_accuracy) / summary_std.regime_accuracy);
    printf("%-25s %12.1f %12.1f %+11.1f%%\n", "Detection Delay (ticks)",
           summary_std.regime_detection_delay, summary_apf.regime_detection_delay,
           100.0 * (summary_std.regime_detection_delay - summary_apf.regime_detection_delay) /
               (summary_std.regime_detection_delay + 1e-10));
    printf("%-25s %12d %12d\n", "False Positives",
           summary_std.false_positives, summary_apf.false_positives);
    printf("%-25s %12d %12d\n", "False Negatives",
           summary_std.false_negatives, summary_apf.false_negatives);
    printf("%-25s %12.1f %12.1f\n", "Avg ESS",
           summary_std.avg_ess, summary_apf.avg_ess);
    printf("%-25s %12.2f %12.2f %+11.1f%%\n", "Avg Latency (μs)",
           summary_std.avg_latency_us, summary_apf.avg_latency_us,
           100.0 * (summary_apf.avg_latency_us - summary_std.avg_latency_us) / summary_std.avg_latency_us);
    printf("%-25s %12.2f %12.2f\n", "P99 Latency (μs)",
           summary_std.p99_latency_us, summary_apf.p99_latency_us);
    printf("=================================================================\n\n");

    /* Scenario-specific breakdown */
    printf("SCENARIO BREAKDOWN (Regime Accuracy):\n");
    printf("-----------------------------------------------------------------\n");
    const char *scenario_names[] = {
        "1. Calm (R0)",
        "2. Gradual increase",
        "3. Sudden crisis",
        "4. Crisis persistence",
        "5. Recovery",
        "6. Flash crash",
        "7. Choppy switching"};

    for (int s = 0; s < data->n_scenarios; s++)
    {
        int start = data->scenario_starts[s];
        int end = (s + 1 < data->n_scenarios) ? data->scenario_starts[s + 1] : n;

        int correct_std = 0, correct_apf = 0;
        for (int t = start; t < end; t++)
        {
            if (records_standard[t].est_regime_smooth == data->true_regime[t])
                correct_std++;
            if (records_apf[t].est_regime_smooth == data->true_regime[t])
                correct_apf++;
        }

        int count = end - start;
        printf("%-25s %11.1f%% %11.1f%%\n", scenario_names[s],
               100.0 * correct_std / count,
               100.0 * correct_apf / count);
    }
    printf("=================================================================\n");

    /* Cleanup */
    free(records_standard);
    free(records_apf);
    free_synthetic_data(data);

    printf("\nDone. Open CSV files in Jupyter for detailed analysis.\n");

    return 0;
}