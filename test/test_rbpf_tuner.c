/**
 * @file test_rbpf_tuner.c
 * @brief Test harness for RBPF parameter auto-tuner
 *
 * Generates synthetic market data with known regimes, then runs
 * the tuner to find optimal RBPF parameters.
 *
 * Usage:
 *   test_rbpf_tuner [fast|normal|detailed]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "rbpf_tuner.h"

/*═══════════════════════════════════════════════════════════════════════════
 * CROSS-PLATFORM TIMING
 *═══════════════════════════════════════════════════════════════════════════*/

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
#include <sys/time.h>
static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * SIMPLE RNG (xoroshiro128+)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    uint64_t s[2];
} SimpleRNG;

static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next(SimpleRNG *rng) {
    uint64_t s0 = rng->s[0];
    uint64_t s1 = rng->s[1];
    uint64_t result = s0 + s1;
    s1 ^= s0;
    rng->s[0] = rotl64(s0, 24) ^ s1 ^ (s1 << 16);
    rng->s[1] = rotl64(s1, 37);
    return result;
}

static void rng_seed(SimpleRNG *rng, uint64_t seed) {
    uint64_t z = seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    rng->s[0] = z ^ (z >> 31);
    z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    rng->s[1] = z ^ (z >> 31);
}

static double rng_uniform(SimpleRNG *rng) {
    return (rng_next(rng) >> 11) * (1.0 / 9007199254740992.0);
}

static double rng_normal(SimpleRNG *rng) {
    /* Box-Muller */
    double u1 = rng_uniform(rng);
    double u2 = rng_uniform(rng);
    while (u1 < 1e-10) u1 = rng_uniform(rng);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

/*═══════════════════════════════════════════════════════════════════════════
 * SYNTHETIC DATA GENERATOR
 *
 * Generates realistic financial returns with:
 *   - 4 volatility regimes (calm, mild, elevated, crisis)
 *   - Markov regime transitions
 *   - Stochastic volatility within each regime
 *   - Optional scheduled events (crisis at specific ticks)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    int n_ticks;
    int n_regimes;
    unsigned int seed;
    
    /* True regime parameters (what we're trying to recover) */
    float true_mu_vol[8];        /* Long-run mean log-volatility per regime */
    float true_sigma_vol[8];     /* Vol-of-vol per regime */
    float true_theta[8];         /* Mean reversion speed per regime */
    
    /* Transition probabilities */
    float trans_matrix[64];      /* [i*n_regimes + j] = P(i -> j) */
    
    /* Scheduled events */
    int crisis_start;            /* Tick when crisis begins (-1 = random) */
    int crisis_duration;         /* How long crisis lasts */
    
} SyntheticConfig;

typedef struct {
    float *returns;              /* [n_ticks] observed returns */
    int   *regimes;              /* [n_ticks] true regime labels */
    float *true_vol;             /* [n_ticks] true volatility */
    float *log_vol;              /* [n_ticks] true log-volatility */
    int    n_ticks;
    
} SyntheticData;

static SyntheticConfig synthetic_config_defaults(void)
{
    SyntheticConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    
    cfg.n_ticks = 8000;
    cfg.n_regimes = 4;
    cfg.seed = 42;
    
    /* True parameters - these define the "ground truth" we're trying to recover
     * Note: Gaps of ~1.5 in log-vol space = factor of ~4.5x in volatility */
    cfg.true_mu_vol[0] = -4.6f;   /* R0: ~1% daily vol (calm) */
    cfg.true_mu_vol[1] = -3.2f;   /* R1: ~4% daily vol (mild) */
    cfg.true_mu_vol[2] = -1.8f;   /* R2: ~17% daily vol (elevated) */
    cfg.true_mu_vol[3] = -0.5f;   /* R3: ~60% daily vol (crisis) */
    
    cfg.true_sigma_vol[0] = 0.10f;
    cfg.true_sigma_vol[1] = 0.15f;
    cfg.true_sigma_vol[2] = 0.20f;
    cfg.true_sigma_vol[3] = 0.25f;
    
    cfg.true_theta[0] = 0.03f;
    cfg.true_theta[1] = 0.05f;
    cfg.true_theta[2] = 0.08f;
    cfg.true_theta[3] = 0.12f;
    
    /* Transition matrix: sticky with adjacent-only transitions */
    float self = 0.985f;
    float off = 1.0f - self;
    
    /* R0: can only go to R1 */
    cfg.trans_matrix[0] = self;
    cfg.trans_matrix[1] = off;
    cfg.trans_matrix[2] = 0.0f;
    cfg.trans_matrix[3] = 0.0f;
    
    /* R1: can go to R0 or R2 */
    cfg.trans_matrix[4] = off/2;
    cfg.trans_matrix[5] = self;
    cfg.trans_matrix[6] = off/2;
    cfg.trans_matrix[7] = 0.0f;
    
    /* R2: can go to R1 or R3 */
    cfg.trans_matrix[8] = 0.0f;
    cfg.trans_matrix[9] = off/2;
    cfg.trans_matrix[10] = self;
    cfg.trans_matrix[11] = off/2;
    
    /* R3: can only go to R2 */
    cfg.trans_matrix[12] = 0.0f;
    cfg.trans_matrix[13] = 0.0f;
    cfg.trans_matrix[14] = off;
    cfg.trans_matrix[15] = self;
    
    /* Scheduled crisis */
    cfg.crisis_start = 2000;
    cfg.crisis_duration = 500;
    
    return cfg;
}

static int synthetic_data_generate(SyntheticData *data, const SyntheticConfig *cfg)
{
    if (!data || !cfg) return -1;
    
    memset(data, 0, sizeof(*data));
    data->n_ticks = cfg->n_ticks;
    
    /* Allocate */
    data->returns = (float*)malloc(cfg->n_ticks * sizeof(float));
    data->regimes = (int*)malloc(cfg->n_ticks * sizeof(int));
    data->true_vol = (float*)malloc(cfg->n_ticks * sizeof(float));
    data->log_vol = (float*)malloc(cfg->n_ticks * sizeof(float));
    
    if (!data->returns || !data->regimes || !data->true_vol || !data->log_vol) {
        free(data->returns);
        free(data->regimes);
        free(data->true_vol);
        free(data->log_vol);
        return -1;
    }
    
    SimpleRNG rng;
    rng_seed(&rng, cfg->seed);
    
    /* Initialize state */
    int regime = 0;
    float log_vol = cfg->true_mu_vol[0];
    
    for (int t = 0; t < cfg->n_ticks; t++) {
        /* Check for scheduled crisis */
        if (cfg->crisis_start > 0) {
            if (t == cfg->crisis_start) {
                regime = cfg->n_regimes - 1;  /* Jump to crisis */
            } else if (t == cfg->crisis_start + cfg->crisis_duration) {
                regime = cfg->n_regimes - 2;  /* Start recovery */
            }
        }
        
        /* Regime transition */
        float u = (float)rng_uniform(&rng);
        float cumsum = 0.0f;
        for (int j = 0; j < cfg->n_regimes; j++) {
            cumsum += cfg->trans_matrix[regime * cfg->n_regimes + j];
            if (u < cumsum) {
                regime = j;
                break;
            }
        }
        
        /* Stochastic volatility evolution */
        float theta = cfg->true_theta[regime];
        float mu = cfg->true_mu_vol[regime];
        float sigma = cfg->true_sigma_vol[regime];
        
        log_vol = (1.0f - theta) * log_vol + theta * mu + 
                  sigma * (float)rng_normal(&rng);
        
        /* Clamp to reasonable range */
        if (log_vol < -8.0f) log_vol = -8.0f;
        if (log_vol > 2.0f) log_vol = 2.0f;
        
        float vol = expf(log_vol);
        
        /* Generate return */
        float ret = vol * (float)rng_normal(&rng);
        
        /* Add occasional jumps in crisis regime */
        if (regime == cfg->n_regimes - 1 && rng_uniform(&rng) < 0.05) {
            ret += (rng_uniform(&rng) < 0.5 ? -1.0f : 1.0f) * vol * 3.0f;
        }
        
        /* Store */
        data->returns[t] = ret;
        data->regimes[t] = regime;
        data->true_vol[t] = vol;
        data->log_vol[t] = log_vol;
    }
    
    return 0;
}

static void synthetic_data_free(SyntheticData *data)
{
    if (!data) return;
    free(data->returns);
    free(data->regimes);
    free(data->true_vol);
    free(data->log_vol);
    memset(data, 0, sizeof(*data));
}

static void synthetic_data_print_stats(const SyntheticData *data, int n_regimes)
{
    if (!data) return;
    
    int regime_counts[8] = {0};
    float regime_vol_sum[8] = {0};
    
    for (int t = 0; t < data->n_ticks; t++) {
        int r = data->regimes[t];
        if (r >= 0 && r < n_regimes) {
            regime_counts[r]++;
            regime_vol_sum[r] += data->true_vol[t];
        }
    }
    
    printf("\nSynthetic Data Statistics:\n");
    printf("  Total ticks: %d\n", data->n_ticks);
    printf("\n  Regime distribution:\n");
    for (int r = 0; r < n_regimes; r++) {
        float avg_vol = regime_counts[r] > 0 ? 
            regime_vol_sum[r] / regime_counts[r] : 0;
        printf("    R%d: %5d ticks (%5.1f%%) - avg vol: %.2f%%\n",
               r, regime_counts[r], 
               100.0f * regime_counts[r] / data->n_ticks,
               avg_vol * 100);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * MANUAL PARAMETER TEST
 *
 * Quick test of specific parameter values before running full search.
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_manual_params(RBPFTuner *tuner)
{
    printf("\n");
    printf("================================================================\n");
    printf("  Manual Parameter Test\n");
    printf("================================================================\n");
    
    int n = tuner->config.n_regimes;
    TunerResult result;
    
    /* Test 1: Wide separation */
    printf("\nTest 1: Wide separation (gap=1.5)\n");
    float mu1[4] = {-5.0f, -3.5f, -2.0f, -0.5f};
    float sig1[4] = {0.08f, 0.10f, 0.12f, 0.15f};
    float th1[4] = {0.03f, 0.05f, 0.07f, 0.10f};
    
    tuner_evaluate_params(tuner, mu1, sig1, th1, 0.98f, 10, 0.75f, &result);
    printf("  Accuracy: %.1f%% (R1=%.1f%%, R2=%.1f%%)\n",
           result.overall_accuracy * 100,
           result.regime_accuracy[1] * 100,
           result.regime_accuracy[2] * 100);
    
    /* Test 2: Very sticky transitions */
    printf("\nTest 2: Very sticky (self=0.995)\n");
    tuner_evaluate_params(tuner, mu1, sig1, th1, 0.995f, 10, 0.75f, &result);
    printf("  Accuracy: %.1f%% (R1=%.1f%%, R2=%.1f%%)\n",
           result.overall_accuracy * 100,
           result.regime_accuracy[1] * 100,
           result.regime_accuracy[2] * 100);
    
    /* Test 3: Strong hysteresis */
    printf("\nTest 3: Strong hysteresis (hold=15, prob=0.85)\n");
    tuner_evaluate_params(tuner, mu1, sig1, th1, 0.98f, 15, 0.85f, &result);
    printf("  Accuracy: %.1f%% (R1=%.1f%%, R2=%.1f%%)\n",
           result.overall_accuracy * 100,
           result.regime_accuracy[1] * 100,
           result.regime_accuracy[2] * 100);
    
    /* Test 4: Combined best guesses */
    printf("\nTest 4: Combined (gap=1.5, self=0.99, hold=12, prob=0.80)\n");
    tuner_evaluate_params(tuner, mu1, sig1, th1, 0.99f, 12, 0.80f, &result);
    printf("  Accuracy: %.1f%% (R1=%.1f%%, R2=%.1f%%)\n",
           result.overall_accuracy * 100,
           result.regime_accuracy[1] * 100,
           result.regime_accuracy[2] * 100);
    tuner_print_confusion(&result, n);
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char *argv[])
{
    printf("\n");
    printf("================================================================\n");
    printf("  RBPF Parameter Auto-Tuner Test\n");
    printf("================================================================\n");
    
    /* Parse command line */
    TunerConfig cfg;
    const char *mode = "normal";
    
    if (argc > 1) {
        mode = argv[1];
    }
    
    if (strcmp(mode, "fast") == 0) {
        cfg = tuner_config_fast();
        printf("  Mode: FAST (reduced grid)\n");
    } else if (strcmp(mode, "detailed") == 0) {
        cfg = tuner_config_detailed();
        printf("  Mode: DETAILED (fine grid)\n");
    } else if (strcmp(mode, "middle") == 0) {
        cfg = tuner_config_focus_middle();
        printf("  Mode: FOCUS MIDDLE (bonus for R1/R2)\n");
    } else {
        cfg = tuner_config_defaults();
        printf("  Mode: NORMAL\n");
    }
    
    /* Generate synthetic data */
    printf("\nGenerating synthetic data...\n");
    
    SyntheticConfig syn_cfg = synthetic_config_defaults();
    SyntheticData data;
    
    if (synthetic_data_generate(&data, &syn_cfg) < 0) {
        fprintf(stderr, "Failed to generate synthetic data\n");
        return 1;
    }
    
    synthetic_data_print_stats(&data, syn_cfg.n_regimes);
    
    /* Print true parameters (what we're trying to recover) */
    printf("\nTrue Parameters (ground truth):\n");
    printf("  %-6s %8s %8s %8s\n", "Regime", "mu_vol", "sigma", "theta");
    for (int r = 0; r < syn_cfg.n_regimes; r++) {
        printf("  R%-5d %8.3f %8.3f %8.3f\n",
               r, syn_cfg.true_mu_vol[r], 
               syn_cfg.true_sigma_vol[r],
               syn_cfg.true_theta[r]);
    }
    
    /* Initialize tuner */
    RBPFTuner tuner;
    if (tuner_init(&tuner, &cfg, data.returns, data.regimes, 
                   data.true_vol, data.n_ticks) < 0) {
        fprintf(stderr, "Failed to initialize tuner\n");
        synthetic_data_free(&data);
        return 1;
    }
    
    /* Run manual tests first */
    test_manual_params(&tuner);
    
    /* Run grid search */
    printf("\n");
    printf("================================================================\n");
    printf("  Starting Grid Search\n");
    printf("================================================================\n");
    
    double t_start = get_time_ms();
    float best_obj = tuner_grid_search(&tuner);
    double elapsed = get_time_ms() - t_start;
    
    printf("\nSearch completed in %.1f seconds\n", elapsed / 1000.0);
    
    /* Print detailed results */
    const TunerResult *best = tuner_get_best(&tuner);
    tuner_print_result(best, cfg.n_regimes);
    tuner_print_confusion(best, cfg.n_regimes);
    
    /* Compare to ground truth */
    printf("\nComparison to Ground Truth:\n");
    printf("  %-6s %10s %10s %8s\n", "Regime", "True mu", "Found mu", "Error");
    for (int r = 0; r < cfg.n_regimes; r++) {
        float error = best->mu_vol[r] - syn_cfg.true_mu_vol[r];
        printf("  R%-5d %10.3f %10.3f %+7.3f\n",
               r, syn_cfg.true_mu_vol[r], best->mu_vol[r], error);
    }
    
    /* Generate code snippet */
    printf("\n");
    printf("================================================================\n");
    printf("  Generated Code\n");
    printf("================================================================\n");
    
    char code_buffer[4096];
    tuner_generate_code(best, cfg.n_regimes, code_buffer, sizeof(code_buffer));
    printf("%s\n", code_buffer);
    
    /* Export to CSV */
    const char *csv_file = "tuner_results.csv";
    if (tuner_export_csv(&tuner, csv_file) == 0) {
        printf("Results exported to: %s\n", csv_file);
    }
    
    /* Cleanup */
    tuner_free(&tuner);
    synthetic_data_free(&data);
    
    printf("\n");
    printf("================================================================\n");
    printf("  Test Complete\n");
    printf("================================================================\n");
    
    return 0;
}
