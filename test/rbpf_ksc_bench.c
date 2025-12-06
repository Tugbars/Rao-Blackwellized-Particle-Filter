/**
 * @file rbpf_ksc_bench.c
 * @brief Benchmark for RBPF-KSC: latency and accuracy vs true volatility
 *
 * Compile:
 *   gcc -O3 -march=native -fopenmp rbpf_ksc.c rbpf_apf.c rbpf_ksc_bench.c -o rbpf_bench \
 *       -lmkl_rt -lm -DNDEBUG
 *
 * Or with Intel compiler:
 *   icx -O3 -xHost -qopenmp rbpf_ksc.c rbpf_apf.c rbpf_ksc_bench.c -o rbpf_bench \
 *       -qmkl -DNDEBUG
 */

#include "rbpf_ksc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

/*─────────────────────────────────────────────────────────────────────────────
 * HIGH-RESOLUTION TIMER
 *───────────────────────────────────────────────────────────────────────────*/

static inline double get_time_us(void)
{
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart * 1e6;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
#endif
}

/*─────────────────────────────────────────────────────────────────────────────
 * SYNTHETIC DATA GENERATOR
 *
 * Generates returns with known stochastic volatility process:
 *   log_vol[t] = (1-θ)*log_vol[t-1] + θ*μ + σ_v*ε_v
 *   return[t] = exp(log_vol[t]) * ε_r
 *
 * With regime switches at known points.
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    rbpf_real_t *returns;      /* Simulated returns */
    rbpf_real_t *true_vol;     /* True volatility (for accuracy measurement) */
    rbpf_real_t *true_log_vol; /* True log-volatility */
    int *true_regime;          /* True regime at each timestep */
    int n;                     /* Number of observations */
} SyntheticData;

static rbpf_real_t randn(void)
{
    /* Box-Muller */
    rbpf_real_t u1 = (rbpf_real_t)rand() / RAND_MAX;
    rbpf_real_t u2 = (rbpf_real_t)rand() / RAND_MAX;
    if (u1 < 1e-10f)
        u1 = 1e-10f;
    return rbpf_sqrt(RBPF_REAL(-2.0) * rbpf_log(u1)) * rbpf_cos(RBPF_REAL(6.283185) * u2);
}

SyntheticData *generate_synthetic_data(int n, int seed)
{
    srand(seed);

    SyntheticData *data = (SyntheticData *)malloc(sizeof(SyntheticData));
    data->n = n;
    data->returns = (rbpf_real_t *)malloc(n * sizeof(rbpf_real_t));
    data->true_vol = (rbpf_real_t *)malloc(n * sizeof(rbpf_real_t));
    data->true_log_vol = (rbpf_real_t *)malloc(n * sizeof(rbpf_real_t));
    data->true_regime = (int *)malloc(n * sizeof(int));

    /* Regime parameters (matching RBPF setup) */
    /* Regime 0: Low vol, stable (~1% daily) */
    /* Regime 1: Medium vol (~3% daily) */
    /* Regime 2: High vol (~8% daily) */
    /* Regime 3: Crisis vol (~20% daily) - must cover peaks! */

    /* NOTE: With sigma_vol noise, the OU process can push log-vol
     * significantly beyond mu_vol. To avoid bias, regime 3's mu_vol
     * should be high enough to "explain" the peak observations. */

    rbpf_real_t theta[4] = {RBPF_REAL(0.05), RBPF_REAL(0.08), RBPF_REAL(0.12), RBPF_REAL(0.15)};
    rbpf_real_t mu_vol[4] = {
        rbpf_log(RBPF_REAL(0.01)), /* -4.6: ~1% vol */
        rbpf_log(RBPF_REAL(0.03)), /* -3.5: ~3% vol */
        rbpf_log(RBPF_REAL(0.08)), /* -2.5: ~8% vol */
        rbpf_log(RBPF_REAL(0.20))  /* -1.6: ~20% vol - covers peaks */
    };
    rbpf_real_t sigma_vol[4] = {RBPF_REAL(0.05), RBPF_REAL(0.10), RBPF_REAL(0.20), RBPF_REAL(0.30)};

    /* Regime schedule: create interesting volatility patterns */
    int regime = 0;
    rbpf_real_t log_vol = mu_vol[0];

    for (int t = 0; t < n; t++)
    {
        /* Regime switches at specific points */
        if (t == n / 5)
            regime = 1; /* 20%: enter medium vol */
        if (t == 2 * n / 5)
            regime = 2; /* 40%: crisis begins */
        if (t == n / 2)
            regime = 3; /* 50%: peak crisis */
        if (t == 3 * n / 5)
            regime = 2; /* 60%: crisis easing */
        if (t == 7 * n / 10)
            regime = 1; /* 70%: recovery */
        if (t == 4 * n / 5)
            regime = 0; /* 80%: back to calm */

        /* Evolve log-vol with OU process */
        rbpf_real_t th = theta[regime];
        rbpf_real_t mv = mu_vol[regime];
        rbpf_real_t sv = sigma_vol[regime];

        log_vol = (RBPF_REAL(1.0) - th) * log_vol + th * mv + sv * randn();

        /* Generate return */
        rbpf_real_t vol = rbpf_exp(log_vol);
        rbpf_real_t ret = vol * randn();

        data->returns[t] = ret;
        data->true_vol[t] = vol;
        data->true_log_vol[t] = log_vol;
        data->true_regime[t] = regime;
    }

    return data;
}

void free_synthetic_data(SyntheticData *data)
{
    if (data)
    {
        free(data->returns);
        free(data->true_vol);
        free(data->true_log_vol);
        free(data->true_regime);
        free(data);
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * ACCURACY METRICS
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    rbpf_real_t mae_vol;         /* Mean Absolute Error on volatility */
    rbpf_real_t rmse_vol;        /* Root Mean Squared Error on volatility */
    rbpf_real_t mae_log_vol;     /* MAE on log-volatility */
    rbpf_real_t max_error_vol;   /* Maximum absolute error */
    rbpf_real_t correlation;     /* Correlation with true vol */
    rbpf_real_t tail_mae;        /* MAE when true_vol > 90th percentile */
    rbpf_real_t regime_accuracy; /* % correct regime detection */
} AccuracyMetrics;

static int compare_real(const void *a, const void *b)
{
    rbpf_real_t fa = *(const rbpf_real_t *)a;
    rbpf_real_t fb = *(const rbpf_real_t *)b;
    return (fa > fb) - (fa < fb);
}

static int compare_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

AccuracyMetrics compute_accuracy(const rbpf_real_t *est_vol, const rbpf_real_t *est_log_vol,
                                 const int *est_regime, const SyntheticData *data)
{
    AccuracyMetrics m = {0};
    int n = data->n;

    /* Basic stats */
    rbpf_real_t sum_ae = 0, sum_se = 0, sum_ae_lv = 0;
    rbpf_real_t max_err = 0;
    int regime_correct = 0;

    /* For correlation */
    rbpf_real_t sum_true = 0, sum_est = 0;
    rbpf_real_t sum_true2 = 0, sum_est2 = 0, sum_te = 0;

    /* Find 90th percentile of true vol for tail analysis */
    rbpf_real_t *sorted_vol = (rbpf_real_t *)malloc(n * sizeof(rbpf_real_t));
    memcpy(sorted_vol, data->true_vol, n * sizeof(rbpf_real_t));
    qsort(sorted_vol, n, sizeof(rbpf_real_t), compare_real);
    rbpf_real_t p90_vol = sorted_vol[(int)(RBPF_REAL(0.9) * n)];
    free(sorted_vol);

    int tail_count = 0;
    rbpf_real_t tail_sum_ae = 0;

    for (int t = 0; t < n; t++)
    {
        rbpf_real_t true_v = data->true_vol[t];
        rbpf_real_t est_v = est_vol[t];
        rbpf_real_t true_lv = data->true_log_vol[t];
        rbpf_real_t est_lv = est_log_vol[t];

        rbpf_real_t ae = rbpf_fabs(true_v - est_v);
        rbpf_real_t se = (true_v - est_v) * (true_v - est_v);
        rbpf_real_t ae_lv = rbpf_fabs(true_lv - est_lv);

        sum_ae += ae;
        sum_se += se;
        sum_ae_lv += ae_lv;
        if (ae > max_err)
            max_err = ae;

        sum_true += true_v;
        sum_est += est_v;
        sum_true2 += true_v * true_v;
        sum_est2 += est_v * est_v;
        sum_te += true_v * est_v;

        if (data->true_regime[t] == est_regime[t])
            regime_correct++;

        /* Tail analysis */
        if (true_v > p90_vol)
        {
            tail_sum_ae += ae;
            tail_count++;
        }
    }

    m.mae_vol = sum_ae / n;
    m.rmse_vol = rbpf_sqrt(sum_se / n);
    m.mae_log_vol = sum_ae_lv / n;
    m.max_error_vol = max_err;
    m.regime_accuracy = (rbpf_real_t)regime_correct / n * RBPF_REAL(100.0);
    m.tail_mae = (tail_count > 0) ? tail_sum_ae / tail_count : 0;

    /* Pearson correlation */
    rbpf_real_t n_f = (rbpf_real_t)n;
    rbpf_real_t cov = sum_te - sum_true * sum_est / n_f;
    rbpf_real_t var_true = sum_true2 - sum_true * sum_true / n_f;
    rbpf_real_t var_est = sum_est2 - sum_est * sum_est / n_f;
    m.correlation = cov / (rbpf_sqrt(var_true * var_est) + 1e-10f);

    return m;
}

/*─────────────────────────────────────────────────────────────────────────────
 * BENCHMARK RUNNER
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    int n_particles;
    double latency_mean_us;
    double latency_p50_us;
    double latency_p99_us;
    double latency_max_us;
    AccuracyMetrics accuracy;
} BenchmarkResult;

BenchmarkResult run_benchmark(int n_particles, const SyntheticData *data,
                              int warmup_iters, int measure_iters)
{
    BenchmarkResult result = {0};
    result.n_particles = n_particles;

    int n_obs = data->n;

    /* Create filter */
    RBPF_KSC *rbpf = rbpf_ksc_create(n_particles, 4);
    if (!rbpf)
    {
        fprintf(stderr, "Failed to create RBPF with %d particles\n", n_particles);
        return result;
    }

    /* Configure regimes to match synthetic data */
    /* Regime params - MUST match synthetic data range! */
    rbpf_ksc_set_regime_params(rbpf, 0, 0.05f, rbpf_log(0.01f), 0.05f); /* -4.6: calm */
    rbpf_ksc_set_regime_params(rbpf, 1, 0.08f, rbpf_log(0.03f), 0.10f); /* -3.5: medium */
    rbpf_ksc_set_regime_params(rbpf, 2, 0.12f, rbpf_log(0.08f), 0.20f); /* -2.5: high */
    rbpf_ksc_set_regime_params(rbpf, 3, 0.15f, rbpf_log(0.20f), 0.30f); /* -1.6: crisis */

    /* Build transition matrix
     *
     * CRITICAL: Transitions must be "loose" enough to allow quick regime changes!
     * Too sticky (0.95 stay prob) → particles get stuck, can't respond to jumps
     * Too loose (0.50 stay prob) → noisy regime estimates
     *
     * Rule of thumb: stay_prob ≈ 0.8-0.9 for HFT with regime changes every 100+ ticks
     */
    rbpf_real_t trans[16] = {
        0.92f, 0.05f, 0.02f, 0.01f, /* From regime 0: sticky */
        0.05f, 0.88f, 0.05f, 0.02f, /* From regime 1 */
        0.02f, 0.05f, 0.88f, 0.05f, /* From regime 2 */
        0.01f, 0.02f, 0.05f, 0.92f  /* From regime 3: sticky */
    };
    rbpf_ksc_build_transition_lut(rbpf, trans);

    /* Regularization */
    rbpf_ksc_set_regularization(rbpf, 0.02f, 0.001f);

    /* Reduced regime diversity mutation to prevent flickering */
    rbpf_ksc_set_regime_diversity(rbpf, n_particles / 25, 0.01f); /* 4% min, 1% mutation */

    /* Allocate output storage */
    rbpf_real_t *est_vol = (rbpf_real_t *)malloc(n_obs * sizeof(rbpf_real_t));
    rbpf_real_t *est_log_vol = (rbpf_real_t *)malloc(n_obs * sizeof(rbpf_real_t));
    int *est_regime = (int *)malloc(n_obs * sizeof(int));
    double *latencies = (double *)malloc(n_obs * sizeof(double));

    /* Warmup runs */
    for (int iter = 0; iter < warmup_iters; iter++)
    {
        rbpf_ksc_init(rbpf, rbpf_log(0.01f), 0.1f);
        rbpf_ksc_warmup(rbpf);

        RBPF_KSC_Output out;
        for (int t = 0; t < n_obs; t++)
        {
            rbpf_ksc_step(rbpf, data->returns[t], &out);
        }
    }

    /* Measurement runs */
    double total_latency = 0;

    for (int iter = 0; iter < measure_iters; iter++)
    {
        rbpf_ksc_init(rbpf, rbpf_log(0.01f), 0.1f);

        RBPF_KSC_Output out;
        for (int t = 0; t < n_obs; t++)
        {
            double t0 = get_time_us();
            rbpf_ksc_step(rbpf, data->returns[t], &out);
            double t1 = get_time_us();

            latencies[t] = t1 - t0;
            total_latency += latencies[t];

            /* Store last iteration's estimates for accuracy */
            if (iter == measure_iters - 1)
            {
                est_vol[t] = out.vol_mean;
                est_log_vol[t] = out.log_vol_mean;
                est_regime[t] = out.dominant_regime;
            }
        }
    }

    /* Latency statistics */
    result.latency_mean_us = total_latency / (measure_iters * n_obs);

    /* Sort for percentiles */
    qsort(latencies, n_obs, sizeof(double), compare_double);
    result.latency_p50_us = latencies[n_obs / 2];
    result.latency_p99_us = latencies[(int)(0.99 * n_obs)];
    result.latency_max_us = latencies[n_obs - 1];

    /* Accuracy */
    result.accuracy = compute_accuracy(est_vol, est_log_vol, est_regime, data);

    /* Cleanup */
    free(est_vol);
    free(est_log_vol);
    free(est_regime);
    free(latencies);
    rbpf_ksc_destroy(rbpf);

    return result;
}

/*─────────────────────────────────────────────────────────────────────────────
 * DIAGNOSTIC OUTPUT
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    /* Fast signal (t) */
    rbpf_real_t *est_vol;
    rbpf_real_t *est_log_vol;
    int *est_regime;

    /* Smooth signal (t-K) */
    rbpf_real_t *est_vol_smooth;
    rbpf_real_t *est_log_vol_smooth;
    int *est_regime_smooth;
    rbpf_real_t *regime_confidence;

    /* Ground truth */
    rbpf_real_t *true_vol;
    rbpf_real_t *true_log_vol;
    int *true_regime;

    /* Self-aware signals */
    rbpf_real_t *surprise;
    rbpf_real_t *regime_entropy;
    rbpf_real_t *vol_ratio;
    rbpf_real_t *ess;
    int *regime_changed;
    int *change_type;
    int n;
} DiagnosticData;

DiagnosticData *run_diagnostic(int n_particles, const SyntheticData *data)
{
    int n_obs = data->n;

    DiagnosticData *diag = (DiagnosticData *)malloc(sizeof(DiagnosticData));
    diag->n = n_obs;

    /* Fast signal (t) */
    diag->est_vol = (rbpf_real_t *)malloc(n_obs * sizeof(rbpf_real_t));
    diag->est_log_vol = (rbpf_real_t *)malloc(n_obs * sizeof(rbpf_real_t));
    diag->est_regime = (int *)malloc(n_obs * sizeof(int));

    /* Smooth signal (t-K) */
    diag->est_vol_smooth = (rbpf_real_t *)malloc(n_obs * sizeof(rbpf_real_t));
    diag->est_log_vol_smooth = (rbpf_real_t *)malloc(n_obs * sizeof(rbpf_real_t));
    diag->est_regime_smooth = (int *)malloc(n_obs * sizeof(int));
    diag->regime_confidence = (rbpf_real_t *)malloc(n_obs * sizeof(rbpf_real_t));

    /* Ground truth */
    diag->true_vol = (rbpf_real_t *)malloc(n_obs * sizeof(rbpf_real_t));
    diag->true_log_vol = (rbpf_real_t *)malloc(n_obs * sizeof(rbpf_real_t));
    diag->true_regime = (int *)malloc(n_obs * sizeof(int));

    /* Self-aware signals */
    diag->surprise = (rbpf_real_t *)malloc(n_obs * sizeof(rbpf_real_t));
    diag->regime_entropy = (rbpf_real_t *)malloc(n_obs * sizeof(rbpf_real_t));
    diag->vol_ratio = (rbpf_real_t *)malloc(n_obs * sizeof(rbpf_real_t));
    diag->ess = (rbpf_real_t *)malloc(n_obs * sizeof(rbpf_real_t));
    diag->regime_changed = (int *)malloc(n_obs * sizeof(int));
    diag->change_type = (int *)malloc(n_obs * sizeof(int));

    /* Copy true values */
    memcpy(diag->true_vol, data->true_vol, n_obs * sizeof(rbpf_real_t));
    memcpy(diag->true_log_vol, data->true_log_vol, n_obs * sizeof(rbpf_real_t));
    memcpy(diag->true_regime, data->true_regime, n_obs * sizeof(int));

    /* Create and configure filter */
    RBPF_KSC *rbpf = rbpf_ksc_create(n_particles, 4);
    if (!rbpf)
        return diag;

    /* Regime parameters must span the full volatility range in the data!
     * If data peaks at log(0.25) ≈ -1.4, regime 3 must have μ_vol near there. */
    rbpf_ksc_set_regime_params(rbpf, 0, 0.05f, rbpf_log(0.01f), 0.05f); /* -4.6: calm */
    rbpf_ksc_set_regime_params(rbpf, 1, 0.08f, rbpf_log(0.03f), 0.10f); /* -3.5: medium */
    rbpf_ksc_set_regime_params(rbpf, 2, 0.12f, rbpf_log(0.08f), 0.20f); /* -2.5: high */
    rbpf_ksc_set_regime_params(rbpf, 3, 0.15f, rbpf_log(0.20f), 0.30f); /* -1.6: crisis */

    /* Transition matrix: sticky regimes to reduce flickering
     *
     * 90% stay probability prevents rapid regime switching.
     * Off-diagonal entries allow gradual transitions (0→1→2→3).
     */
    rbpf_real_t trans[16] = {
        0.92f, 0.05f, 0.02f, 0.01f, /* From regime 0: mostly stay */
        0.05f, 0.88f, 0.05f, 0.02f, /* From regime 1 */
        0.02f, 0.05f, 0.88f, 0.05f, /* From regime 2 */
        0.01f, 0.02f, 0.05f, 0.92f  /* From regime 3: mostly stay */
    };
    rbpf_ksc_build_transition_lut(rbpf, trans);
    rbpf_ksc_set_regularization(rbpf, 0.02f, 0.001f);

    /* Reduce regime diversity mutation to prevent noise
     * Keep minimum particles but lower mutation rate */
    rbpf_ksc_set_regime_diversity(rbpf, n_particles / 25, 0.01f); /* 4% min, 1% mutation */

    /* Regime smoothing: require 8 consecutive ticks OR 75% probability to switch */
    rbpf_ksc_set_regime_smoothing(rbpf, 8, 0.75f);

    /* Fixed-lag smoothing: 5-tick delay for regime confirmation
     * Fast signal (t): immediate reaction to volatility spikes
     * Smooth signal (t-5): stable regime for position sizing */
    rbpf_ksc_set_fixed_lag_smoothing(rbpf, 5);

    rbpf_ksc_init(rbpf, rbpf_log(0.01f), 0.1f);

    /* Run filter and collect diagnostics */
    RBPF_KSC_Output out;
    for (int t = 0; t < n_obs; t++)
    {
        rbpf_ksc_step(rbpf, data->returns[t], &out);

        /* Fast signal (t) - immediate */
        diag->est_vol[t] = out.vol_mean;
        diag->est_log_vol[t] = out.log_vol_mean;
        diag->est_regime[t] = out.dominant_regime; /* Fast: instantaneous */

        /* Smooth signal (t-K) - delayed */
        diag->est_vol_smooth[t] = out.vol_mean_smooth;
        diag->est_log_vol_smooth[t] = out.log_vol_mean_smooth;
        diag->est_regime_smooth[t] = out.dominant_regime_smooth; /* Smooth: K-lagged */
        diag->regime_confidence[t] = out.regime_confidence;

        /* Self-aware signals */
        diag->surprise[t] = out.surprise;
        diag->regime_entropy[t] = out.regime_entropy;
        diag->vol_ratio[t] = out.vol_ratio;
        diag->ess[t] = out.ess;
        diag->regime_changed[t] = out.regime_changed;
        diag->change_type[t] = out.change_type;
    }

    rbpf_ksc_destroy(rbpf);
    return diag;
}

void free_diagnostic(DiagnosticData *diag)
{
    if (diag)
    {
        /* Fast signal */
        free(diag->est_vol);
        free(diag->est_log_vol);
        free(diag->est_regime);

        /* Smooth signal */
        free(diag->est_vol_smooth);
        free(diag->est_log_vol_smooth);
        free(diag->est_regime_smooth);
        free(diag->regime_confidence);

        /* Ground truth */
        free(diag->true_vol);
        free(diag->true_log_vol);
        free(diag->true_regime);

        /* Self-aware signals */
        free(diag->surprise);
        free(diag->regime_entropy);
        free(diag->vol_ratio);
        free(diag->ess);
        free(diag->regime_changed);
        free(diag->change_type);
        free(diag);
    }
}

void export_diagnostic_csv(const DiagnosticData *diag, const char *filename)
{
    FILE *f = fopen(filename, "w");
    if (!f)
    {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }

    /* Header includes both fast and smooth signals */
    fprintf(f, "t,true_vol,est_vol_fast,est_vol_smooth,"
               "true_log_vol,est_log_vol_fast,est_log_vol_smooth,"
               "true_regime,est_regime_fast,est_regime_smooth,regime_confidence,"
               "surprise,entropy,vol_ratio,ess,detected,change_type\n");

    for (int t = 0; t < diag->n; t++)
    {
        fprintf(f, "%d,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%d,%d,%d,%.3f,%.4f,%.4f,%.4f,%.1f,%d,%d\n",
                t,
                diag->true_vol[t], diag->est_vol[t], diag->est_vol_smooth[t],
                diag->true_log_vol[t], diag->est_log_vol[t], diag->est_log_vol_smooth[t],
                diag->true_regime[t], diag->est_regime[t], diag->est_regime_smooth[t],
                diag->regime_confidence[t],
                diag->surprise[t], diag->regime_entropy[t],
                diag->vol_ratio[t], diag->ess[t],
                diag->regime_changed[t], diag->change_type[t]);
    }

    fclose(f);
    printf("Diagnostic data exported to: %s\n", filename);
}

void print_changepoint_analysis(const DiagnosticData *diag, const SyntheticData *data)
{
    printf("\n=== Changepoint Analysis ===\n\n");

    /* Find true changepoints */
    int n = data->n;
    int prev_regime = data->true_regime[0];

    printf("True Changepoints vs Filter Response:\n");
    printf("%-8s %-12s %-12s %-10s %-10s %-10s %-10s\n",
           "Time", "Regime", "Est Regime", "Surprise", "Entropy", "VolRatio", "Detected?");
    printf("────────────────────────────────────────────────────────────────────────────\n");

    for (int t = 1; t < n; t++)
    {
        if (data->true_regime[t] != prev_regime)
        {
            /* True changepoint - show window around it */
            int detected_nearby = 0;
            rbpf_real_t max_surprise = 0;
            rbpf_real_t max_entropy = 0;

            /* Look in window [t-5, t+10] for detection */
            for (int w = (t > 5 ? t - 5 : 0); w < (t + 10 < n ? t + 10 : n); w++)
            {
                if (diag->regime_changed[w])
                    detected_nearby = 1;
                if (diag->surprise[w] > max_surprise)
                    max_surprise = diag->surprise[w];
                if (diag->regime_entropy[w] > max_entropy)
                    max_entropy = diag->regime_entropy[w];
            }

            printf("%-8d %d → %-8d %-12d %-10.2f %-10.2f %-10.2f %-10s\n",
                   t,
                   prev_regime, data->true_regime[t],
                   diag->est_regime[t],
                   diag->surprise[t],
                   diag->regime_entropy[t],
                   diag->vol_ratio[t],
                   detected_nearby ? "✓" : "✗");

            prev_regime = data->true_regime[t];
        }
    }

    /* Count false positives */
    int false_positives = 0;
    int true_positives = 0;
    prev_regime = data->true_regime[0];

    for (int t = 1; t < n; t++)
    {
        if (diag->regime_changed[t])
        {
            /* Check if near a true changepoint */
            int near_true_cp = 0;
            for (int w = (t > 10 ? t - 10 : 0); w < (t + 10 < n ? t + 10 : n); w++)
            {
                if (w > 0 && data->true_regime[w] != data->true_regime[w - 1])
                {
                    near_true_cp = 1;
                    break;
                }
            }
            if (near_true_cp)
                true_positives++;
            else
                false_positives++;
        }
    }

    printf("\nDetection Summary:\n");
    printf("  True positives:  %d\n", true_positives);
    printf("  False positives: %d\n", false_positives);
}

void print_tracking_summary(const DiagnosticData *diag)
{
    printf("\n=== Tracking Quality by Regime ===\n\n");

    int n = diag->n;
    rbpf_real_t mae_by_regime[4] = {0};
    int count_by_regime[4] = {0};

    for (int t = 0; t < n; t++)
    {
        int r = diag->true_regime[t];
        if (r >= 0 && r < 4)
        {
            mae_by_regime[r] += rbpf_fabs(diag->true_log_vol[t] - diag->est_log_vol[t]);
            count_by_regime[r]++;
        }
    }

    printf("%-10s %-10s %-12s %-12s\n", "Regime", "Count", "MAE(log-vol)", "Typical Vol");
    printf("────────────────────────────────────────────────\n");

    rbpf_real_t typical_vol[4] = {RBPF_REAL(0.01), RBPF_REAL(0.03), RBPF_REAL(0.08), RBPF_REAL(0.20)};
    for (int r = 0; r < 4; r++)
    {
        if (count_by_regime[r] > 0)
        {
            mae_by_regime[r] /= count_by_regime[r];
        }
        printf("%-10d %-10d %-12.4f %-12.4f\n",
               r, count_by_regime[r], mae_by_regime[r], typical_vol[r]);
    }

    /* Surprise statistics */
    printf("\n=== Surprise Statistics ===\n\n");

    rbpf_real_t surprise_sum = 0, surprise_max = 0;
    for (int t = 0; t < n; t++)
    {
        surprise_sum += diag->surprise[t];
        if (diag->surprise[t] > surprise_max)
            surprise_max = diag->surprise[t];
    }

    printf("Mean surprise: %.2f\n", surprise_sum / n);
    printf("Max surprise:  %.2f\n", surprise_max);

    /* ESS statistics */
    rbpf_real_t ess_sum = 0, ess_min = diag->ess[0];
    for (int t = 0; t < n; t++)
    {
        ess_sum += diag->ess[t];
        if (diag->ess[t] < ess_min)
            ess_min = diag->ess[t];
    }

    printf("\nMean ESS: %.1f\n", ess_sum / n);
    printf("Min ESS:  %.1f\n", ess_min);

    /* Fast vs Smooth Regime Comparison */
    printf("\n=== Fast vs Smooth Regime Accuracy ===\n\n");

    int fast_correct = 0, smooth_correct = 0;
    int smooth_stable = 0; /* Count where smooth != fast (smoothing active) */

    for (int t = 0; t < n; t++)
    {
        if (diag->est_regime[t] == diag->true_regime[t])
            fast_correct++;
        if (diag->est_regime_smooth[t] == diag->true_regime[t])
            smooth_correct++;
        if (diag->est_regime[t] != diag->est_regime_smooth[t])
            smooth_stable++;
    }

    printf("Fast regime accuracy:   %.1f%% (instantaneous)\n", 100.0f * fast_correct / n);
    printf("Smooth regime accuracy: %.1f%% (5-tick lag)\n", 100.0f * smooth_correct / n);
    printf("Smoothing difference:   %.1f%% of ticks\n", 100.0f * smooth_stable / n);
}

/*─────────────────────────────────────────────────────────────────────────────
 * MAIN
 *───────────────────────────────────────────────────────────────────────────*/

int main(int argc, char **argv)
{
    printf("=== RBPF-KSC Benchmark ===\n\n");

    /* Configuration */
    int n_obs = 5000; /* Observations per run */
    int warmup = 3;   /* Warmup iterations */
    int measure = 5;  /* Measurement iterations */
    int seed = 42;

    /* Particle counts to test */
    int particle_counts[] = {50, 100, 200, 500, 1000, 2000};
    int n_configs = sizeof(particle_counts) / sizeof(particle_counts[0]);

    /* Generate synthetic data */
    printf("Generating synthetic data: %d observations, seed=%d\n", n_obs, seed);
    SyntheticData *data = generate_synthetic_data(n_obs, seed);

    /* Print data summary */
    rbpf_real_t min_vol = data->true_vol[0], max_vol = data->true_vol[0];
    for (int t = 1; t < n_obs; t++)
    {
        if (data->true_vol[t] < min_vol)
            min_vol = data->true_vol[t];
        if (data->true_vol[t] > max_vol)
            max_vol = data->true_vol[t];
    }
    printf("True volatility range: [%.4f, %.4f] (%.1fx dynamic range)\n\n",
           min_vol, max_vol, max_vol / min_vol);

    /* Run benchmarks */
    printf("%-10s | %8s %8s %8s %8s | %8s %8s %8s %6s\n",
           "Particles", "Mean", "P50", "P99", "Max",
           "MAE_vol", "Tail_MAE", "Corr", "Regime%");
    printf("%-10s | %8s %8s %8s %8s | %8s %8s %8s %6s\n",
           "", "(μs)", "(μs)", "(μs)", "(μs)", "", "", "", "");
    printf("──────────────────────────────────────────────────────────────────────────\n");

    for (int i = 0; i < n_configs; i++)
    {
        int n_p = particle_counts[i];
        BenchmarkResult r = run_benchmark(n_p, data, warmup, measure);

        printf("%-10d | %8.2f %8.2f %8.2f %8.2f | %8.4f %8.4f %8.4f %5.1f%%\n",
               r.n_particles,
               r.latency_mean_us, r.latency_p50_us, r.latency_p99_us, r.latency_max_us,
               r.accuracy.mae_vol, r.accuracy.tail_mae,
               r.accuracy.correlation, r.accuracy.regime_accuracy);
    }

    printf("\n");

    /* Detailed report for recommended config */
    printf("=== Detailed Report (n=200 particles) ===\n");
    BenchmarkResult detailed = run_benchmark(200, data, warmup, measure);
    printf("Latency:\n");
    printf("  Mean:  %.2f μs\n", detailed.latency_mean_us);
    printf("  P50:   %.2f μs\n", detailed.latency_p50_us);
    printf("  P99:   %.2f μs\n", detailed.latency_p99_us);
    printf("  Max:   %.2f μs\n", detailed.latency_max_us);
    printf("\nAccuracy:\n");
    printf("  MAE (vol):      %.4f\n", detailed.accuracy.mae_vol);
    printf("  RMSE (vol):     %.4f\n", detailed.accuracy.rmse_vol);
    printf("  MAE (log-vol):  %.4f\n", detailed.accuracy.mae_log_vol);
    printf("  Max error:      %.4f\n", detailed.accuracy.max_error_vol);
    printf("  Tail MAE:       %.4f (90th percentile vol)\n", detailed.accuracy.tail_mae);
    printf("  Correlation:    %.4f\n", detailed.accuracy.correlation);
    printf("  Regime acc:     %.1f%%\n", detailed.accuracy.regime_accuracy);

    /* Budget check */
    printf("\n=== Latency Budget Check (200μs total) ===\n");
    rbpf_real_t ssa_us = RBPF_REAL(140.0);
    rbpf_real_t rbpf_us = (rbpf_real_t)detailed.latency_mean_us;
    rbpf_real_t kelly_us = RBPF_REAL(0.5);
    rbpf_real_t total_us = ssa_us + rbpf_us + kelly_us;
    rbpf_real_t headroom = RBPF_REAL(200.0) - total_us;

    printf("  SSA:      %.1f μs\n", ssa_us);
    printf("  RBPF:     %.1f μs (measured)\n", rbpf_us);
    printf("  Kelly:    %.1f μs\n", kelly_us);
    printf("  ─────────────────\n");
    printf("  Total:    %.1f μs\n", total_us);
    printf("  Headroom: %.1f μs %s\n", headroom,
           headroom > 0 ? "✓" : "✗ OVER BUDGET");

    /* Run diagnostics with 200 particles */
    printf("\n");
    printf("════════════════════════════════════════════════════════════════════════════\n");
    printf("                         DIAGNOSTIC ANALYSIS\n");
    printf("════════════════════════════════════════════════════════════════════════════\n");

    DiagnosticData *diag = run_diagnostic(200, data);

    print_changepoint_analysis(diag, data);
    print_tracking_summary(diag);

    /* Export CSV for plotting */
    export_diagnostic_csv(diag, "rbpf_diagnostic.csv");

    printf("\n=== Plotting Commands ===\n");
    printf("Python:\n");
    printf("  import pandas as pd\n");
    printf("  import matplotlib.pyplot as plt\n");
    printf("  df = pd.read_csv('rbpf_diagnostic.csv')\n");
    printf("  \n");
    printf("  fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)\n");
    printf("  \n");
    printf("  # Log-vol tracking (fast vs smooth)\n");
    printf("  axes[0].plot(df['t'], df['true_log_vol'], 'b-', alpha=0.7, label='True')\n");
    printf("  axes[0].plot(df['t'], df['est_log_vol_fast'], 'r-', alpha=0.5, label='Fast (t)')\n");
    printf("  axes[0].plot(df['t'], df['est_log_vol_smooth'], 'g-', alpha=0.7, label='Smooth (t-5)')\n");
    printf("  axes[0].set_ylabel('Log-Vol')\n");
    printf("  axes[0].legend()\n");
    printf("  \n");
    printf("  # Regime (fast vs smooth)\n");
    printf("  axes[1].plot(df['t'], df['true_regime'], 'b-', lw=2, label='True')\n");
    printf("  axes[1].plot(df['t'], df['est_regime_fast'], 'r--', alpha=0.5, label='Fast')\n");
    printf("  axes[1].plot(df['t'], df['est_regime_smooth'], 'g-', alpha=0.7, label='Smooth')\n");
    printf("  axes[1].set_ylabel('Regime')\n");
    printf("  axes[1].legend()\n");
    printf("  \n");
    printf("  # Regime confidence\n");
    printf("  axes[2].plot(df['t'], df['regime_confidence'], 'purple')\n");
    printf("  axes[2].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='70%% threshold')\n");
    printf("  axes[2].set_ylabel('Confidence')\n");
    printf("  axes[2].set_ylim([0, 1])\n");
    printf("  \n");
    printf("  # Surprise\n");
    printf("  axes[3].plot(df['t'], df['surprise'], 'orange')\n");
    printf("  axes[3].axhline(y=5, color='r', linestyle='--', alpha=0.5)\n");
    printf("  axes[3].set_ylabel('Surprise')\n");
    printf("  \n");
    printf("  # ESS\n");
    printf("  axes[4].plot(df['t'], df['ess'], 'teal')\n");
    printf("  axes[4].axhline(y=100, color='r', linestyle='--', alpha=0.5)\n");
    printf("  axes[4].set_ylabel('ESS')\n");
    printf("  axes[4].set_xlabel('Time')\n");
    printf("  \n");
    printf("  plt.tight_layout()\n");
    printf("  plt.savefig('rbpf_diagnostic.png', dpi=150)\n");
    printf("  plt.show()\n");

    free_diagnostic(diag);

    /* ═══════════════════════════════════════════════════════════════════════
     * APF (AUXILIARY PARTICLE FILTER) COMPARISON
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("\n");
    printf("════════════════════════════════════════════════════════════════════════════\n");
    printf("                    APF vs SIR COMPARISON (200 particles)\n");
    printf("════════════════════════════════════════════════════════════════════════════\n\n");

    {
        int n_p = 200;
        int n_obs_apf = data->n;

        /* Create filter for APF test */
        RBPF_KSC *rbpf_apf = rbpf_ksc_create(n_p, 4);
        if (rbpf_apf)
        {
            /* Configure same as main benchmark */
            rbpf_ksc_set_regime_params(rbpf_apf, 0, 0.05f, rbpf_log(0.01f), 0.05f);
            rbpf_ksc_set_regime_params(rbpf_apf, 1, 0.08f, rbpf_log(0.03f), 0.10f);
            rbpf_ksc_set_regime_params(rbpf_apf, 2, 0.12f, rbpf_log(0.08f), 0.20f);
            rbpf_ksc_set_regime_params(rbpf_apf, 3, 0.15f, rbpf_log(0.20f), 0.30f);

            rbpf_real_t trans_apf[16] = {
                0.92f, 0.05f, 0.02f, 0.01f,
                0.05f, 0.88f, 0.05f, 0.02f,
                0.02f, 0.05f, 0.88f, 0.05f,
                0.01f, 0.02f, 0.05f, 0.92f};
            rbpf_ksc_build_transition_lut(rbpf_apf, trans_apf);
            rbpf_ksc_set_regularization(rbpf_apf, 0.02f, 0.001f);
            rbpf_ksc_set_regime_diversity(rbpf_apf, n_p / 25, 0.01f);
            rbpf_ksc_set_fixed_lag_smoothing(rbpf_apf, 5);

            /* Allocate storage */
            rbpf_real_t *est_vol_sir = (rbpf_real_t *)malloc(n_obs_apf * sizeof(rbpf_real_t));
            rbpf_real_t *est_vol_apf = (rbpf_real_t *)malloc(n_obs_apf * sizeof(rbpf_real_t));
            rbpf_real_t *est_vol_adaptive = (rbpf_real_t *)malloc(n_obs_apf * sizeof(rbpf_real_t));
            rbpf_real_t *est_log_vol_sir = (rbpf_real_t *)malloc(n_obs_apf * sizeof(rbpf_real_t));
            rbpf_real_t *est_log_vol_apf = (rbpf_real_t *)malloc(n_obs_apf * sizeof(rbpf_real_t));
            rbpf_real_t *est_log_vol_adaptive = (rbpf_real_t *)malloc(n_obs_apf * sizeof(rbpf_real_t));
            int *est_regime_sir = (int *)malloc(n_obs_apf * sizeof(int));
            int *est_regime_apf = (int *)malloc(n_obs_apf * sizeof(int));
            int *est_regime_adaptive = (int *)malloc(n_obs_apf * sizeof(int));

            double total_sir_us = 0, total_apf_us = 0, total_adaptive_us = 0;
            int apf_trigger_count = 0;

            /* ─── Run 1: Standard SIR ─── */
            rbpf_ksc_init(rbpf_apf, rbpf_log(0.01f), 0.1f);
            rbpf_ksc_warmup(rbpf_apf);

            RBPF_KSC_Output out;
            for (int t = 0; t < n_obs_apf; t++)
            {
                double t0 = get_time_us();
                rbpf_ksc_step(rbpf_apf, data->returns[t], &out);
                double t1 = get_time_us();
                total_sir_us += (t1 - t0);

                est_vol_sir[t] = out.vol_mean;
                est_log_vol_sir[t] = out.log_vol_mean;
                est_regime_sir[t] = out.dominant_regime;
            }

            /* ─── Run 2: Always APF ─── */
            rbpf_ksc_init(rbpf_apf, rbpf_log(0.01f), 0.1f);

            for (int t = 0; t < n_obs_apf - 1; t++)
            {
                double t0 = get_time_us();
                rbpf_ksc_step_apf(rbpf_apf, data->returns[t], data->returns[t + 1], &out);
                double t1 = get_time_us();
                total_apf_us += (t1 - t0);

                est_vol_apf[t] = out.vol_mean;
                est_log_vol_apf[t] = out.log_vol_mean;
                est_regime_apf[t] = out.dominant_regime;
            }
            /* Last tick: no lookahead available */
            rbpf_ksc_step(rbpf_apf, data->returns[n_obs_apf - 1], &out);
            est_vol_apf[n_obs_apf - 1] = out.vol_mean;
            est_log_vol_apf[n_obs_apf - 1] = out.log_vol_mean;
            est_regime_apf[n_obs_apf - 1] = out.dominant_regime;

            /* ─── Run 3: Adaptive APF ─── */
            rbpf_ksc_init(rbpf_apf, rbpf_log(0.01f), 0.1f);

            for (int t = 0; t < n_obs_apf - 1; t++)
            {
                double t0 = get_time_us();
                rbpf_ksc_step_adaptive(rbpf_apf, data->returns[t], data->returns[t + 1], &out);
                double t1 = get_time_us();
                total_adaptive_us += (t1 - t0);

                if (out.apf_triggered)
                    apf_trigger_count++;

                est_vol_adaptive[t] = out.vol_mean;
                est_log_vol_adaptive[t] = out.log_vol_mean;
                est_regime_adaptive[t] = out.dominant_regime;
            }
            /* Last tick */
            rbpf_ksc_step(rbpf_apf, data->returns[n_obs_apf - 1], &out);
            est_vol_adaptive[n_obs_apf - 1] = out.vol_mean;
            est_log_vol_adaptive[n_obs_apf - 1] = out.log_vol_mean;
            est_regime_adaptive[n_obs_apf - 1] = out.dominant_regime;

            /* Compute accuracy for each */
            AccuracyMetrics acc_sir = compute_accuracy(est_vol_sir, est_log_vol_sir, est_regime_sir, data);
            AccuracyMetrics acc_apf = compute_accuracy(est_vol_apf, est_log_vol_apf, est_regime_apf, data);
            AccuracyMetrics acc_adaptive = compute_accuracy(est_vol_adaptive, est_log_vol_adaptive, est_regime_adaptive, data);

            /* Report */
            printf("%-12s | %8s | %8s %8s %8s %6s\n",
                   "Method", "Latency", "MAE_vol", "Tail_MAE", "Corr", "Regime%");
            printf("─────────────────────────────────────────────────────────────────\n");

            printf("%-12s | %6.2f μs | %8.4f %8.4f %8.4f %5.1f%%\n",
                   "SIR",
                   total_sir_us / n_obs_apf,
                   acc_sir.mae_vol, acc_sir.tail_mae, acc_sir.correlation, acc_sir.regime_accuracy);

            printf("%-12s | %6.2f μs | %8.4f %8.4f %8.4f %5.1f%%\n",
                   "APF (always)",
                   total_apf_us / (n_obs_apf - 1),
                   acc_apf.mae_vol, acc_apf.tail_mae, acc_apf.correlation, acc_apf.regime_accuracy);

            printf("%-12s | %6.2f μs | %8.4f %8.4f %8.4f %5.1f%%\n",
                   "Adaptive",
                   total_adaptive_us / (n_obs_apf - 1),
                   acc_adaptive.mae_vol, acc_adaptive.tail_mae, acc_adaptive.correlation, acc_adaptive.regime_accuracy);

            printf("\nAdaptive APF Statistics:\n");
            printf("  APF triggered: %d / %d ticks (%.1f%%)\n",
                   apf_trigger_count, n_obs_apf - 1,
                   100.0 * apf_trigger_count / (n_obs_apf - 1));
            printf("  Avg latency when triggered: ~%.1f μs (vs ~%.1f μs SIR)\n",
                   total_apf_us / (n_obs_apf - 1),
                   total_sir_us / n_obs_apf);

            /* Improvement summary */
            rbpf_real_t regime_improvement = acc_apf.regime_accuracy - acc_sir.regime_accuracy;
            rbpf_real_t tail_improvement = (acc_sir.tail_mae - acc_apf.tail_mae) / acc_sir.tail_mae * 100;

            printf("\nAPF vs SIR Improvement:\n");
            printf("  Regime accuracy: %+.1f%%\n", regime_improvement);
            printf("  Tail MAE:        %+.1f%% %s\n", tail_improvement,
                   tail_improvement > 0 ? "(better)" : "(worse)");

            /* Cleanup */
            free(est_vol_sir);
            free(est_vol_apf);
            free(est_vol_adaptive);
            free(est_log_vol_sir);
            free(est_log_vol_apf);
            free(est_log_vol_adaptive);
            free(est_regime_sir);
            free(est_regime_apf);
            free(est_regime_adaptive);
            rbpf_ksc_destroy(rbpf_apf);
        }
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * LIU-WEST PARAMETER LEARNING TEST
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("\n");
    printf("════════════════════════════════════════════════════════════════════════════\n");
    printf("                    LIU-WEST PARAMETER LEARNING TEST\n");
    printf("════════════════════════════════════════════════════════════════════════════\n\n");

    /* Create filter with wrong initial parameters */
    RBPF_KSC *rbpf_lw = rbpf_ksc_create(200, 4);
    if (rbpf_lw)
    {
        /* Set WRONG initial parameters (deliberately off - small bidirectional errors)
         *
         * Initial values are ORDERED (μ_0 < μ_1 < μ_2 < μ_3) to match order constraint.
         * Errors are small (~0.3-0.5 in log scale) and bidirectional.
         */
        printf("Initial (wrong) parameters:\n");
        printf("  Regime 0: μ_vol = %.4f (true: %.4f) [err=+0.41]\n", rbpf_log(0.015f), rbpf_log(0.01f));
        printf("  Regime 1: μ_vol = %.4f (true: %.4f) [err=-0.18]\n", rbpf_log(0.025f), rbpf_log(0.03f));
        printf("  Regime 2: μ_vol = %.4f (true: %.4f) [err=-0.29]\n", rbpf_log(0.06f), rbpf_log(0.08f));
        printf("  Regime 3: μ_vol = %.4f (true: %.4f) [err=-0.29]\n", rbpf_log(0.15f), rbpf_log(0.20f));

        rbpf_ksc_set_regime_params(rbpf_lw, 0, 0.05f, rbpf_log(0.015f), 0.05f); /* Too high */
        rbpf_ksc_set_regime_params(rbpf_lw, 1, 0.08f, rbpf_log(0.025f), 0.10f); /* Too low */
        rbpf_ksc_set_regime_params(rbpf_lw, 2, 0.15f, rbpf_log(0.06f), 0.20f);  /* Too low */
        rbpf_ksc_set_regime_params(rbpf_lw, 3, 0.20f, rbpf_log(0.15f), 0.30f);  /* Too low */

        rbpf_real_t trans[16] = {
            0.85f, 0.08f, 0.05f, 0.02f,
            0.08f, 0.80f, 0.08f, 0.04f,
            0.04f, 0.08f, 0.80f, 0.08f,
            0.02f, 0.05f, 0.08f, 0.85f};
        rbpf_ksc_build_transition_lut(rbpf_lw, trans);
        rbpf_ksc_set_regularization(rbpf_lw, 0.02f, 0.001f);
        rbpf_ksc_set_regime_diversity(rbpf_lw, 25, 0.03f); /* 25 min per regime, 3% mutation */

        /* Enable Liu-West learning with settings tuned for convergence
         *
         * Key insight: Liu-West only learns during resample!
         * - Shrinkage 0.92: Faster adaptation than 0.95
         * - Warmup 30: Balance between stability and learning
         * - ESS threshold 0.80: Resample frequently but not too often
         * - Force resample every 5 ticks: Guarantee learning happens
         *
         * Order constraint prevents label switching (regime 0 always lowest vol)
         */
        rbpf_ksc_enable_liu_west(rbpf_lw, 0.92f, 30);      /* a=0.92 (faster), warmup=30 */
        rbpf_ksc_set_liu_west_resample(rbpf_lw, 0.80f, 5); /* ESS<80%, force every 5 ticks */
        rbpf_ksc_set_liu_west_bounds(rbpf_lw,
                                     rbpf_log(0.001f), rbpf_log(RBPF_REAL(0.4)), /* μ_vol: [-6.9, -0.9] */
                                     RBPF_REAL(0.01), RBPF_REAL(0.6));           /* σ_vol bounds */

        rbpf_ksc_init(rbpf_lw, rbpf_log(0.01f), 0.1f);

        /* Run filter on synthetic data */
        RBPF_KSC_Output out;
        int n_obs = data->n;

        /* Track learned params over time */
        rbpf_real_t learned_mu_vol[4][5]; /* 4 regimes × 5 checkpoints */
        int checkpoints[] = {100, 500, 1000, 2500, 5000};
        int cp_idx = 0;

        for (int t = 0; t < n_obs; t++)
        {
            rbpf_ksc_step(rbpf_lw, data->returns[t], &out);

            /* Record at checkpoints */
            if (cp_idx < 5 && t == checkpoints[cp_idx] - 1)
            {
                for (int r = 0; r < 4; r++)
                {
                    learned_mu_vol[r][cp_idx] = out.learned_mu_vol[r];
                }
                cp_idx++;
            }
        }

        /* Report learning progress */
        printf("\nLearned μ_vol over time:\n");
        printf("%-8s", "Regime");
        for (int c = 0; c < 5; c++)
        {
            printf("  t=%-5d", checkpoints[c]);
        }
        printf("  True\n");
        printf("────────────────────────────────────────────────────────────────\n");

        rbpf_real_t true_mu_vol[4] = {
            rbpf_log(RBPF_REAL(0.01)), /* -4.6 */
            rbpf_log(RBPF_REAL(0.03)), /* -3.5 */
            rbpf_log(RBPF_REAL(0.08)), /* -2.5 */
            rbpf_log(RBPF_REAL(0.20))  /* -1.6 */
        };
        for (int r = 0; r < 4; r++)
        {
            printf("%-8d", r);
            for (int c = 0; c < 5; c++)
            {
                printf("  %7.3f", learned_mu_vol[r][c]);
            }
            printf("  %7.3f\n", true_mu_vol[r]);
        }

        /* Compute final error */
        printf("\nFinal learning error (|learned - true|):\n");
        rbpf_real_t total_error = 0;
        for (int r = 0; r < 4; r++)
        {
            rbpf_real_t error = rbpf_fabs(learned_mu_vol[r][4] - true_mu_vol[r]);
            total_error += error;
            printf("  Regime %d: %.4f\n", r, error);
        }
        printf("  Total:    %.4f\n", total_error);

        if (total_error < RBPF_REAL(1.0))
        {
            printf("\n✓ Liu-West learning PASSED (error < 1.0)\n");
        }
        else
        {
            printf("\n✗ Liu-West learning needs tuning (error = %.2f)\n", total_error);
        }

        rbpf_ksc_destroy(rbpf_lw);
    }

    /* Cleanup */
    free_synthetic_data(data);

    return 0;
}