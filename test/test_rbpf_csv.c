/**
 * @file test_rbpf_csv.c
 * @brief RBPF validation test with CSV output for Jupyter visualization
 *
 * Purpose: Generate detailed per-tick data to visualize:
 *   1. How accurately RBPF tracks true log-volatility
 *   2. How quickly and accurately it detects regime changes
 *   3. Whether it overreacts to outliers vs follows real changes
 *
 * Output: CSV files in ../../RBPF/csv/ (relative to build/RBPF/)
 *
 * Compile: Part of CMakeLists.txt
 * Run: ./test_rbpf_csv.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

#include "rbpf_ksc.h"
#include "rbpf_ksc_param_integration.h"

/*============================================================================
 * CONFIGURATION
 *============================================================================*/

/* Path works when calling from RBPF/ directory */
#define CSV_OUTPUT_DIR "csv/"
#define N_PARTICLES 256
#define N_REGIMES 4

/* Random seed for reproducibility */
#define RANDOM_SEED 12345

/*============================================================================
 * TEST PHASES
 *
 * Each phase tests a specific aspect of filter performance
 *============================================================================*/

typedef struct
{
    const char *name;
    int start_tick;
    int end_tick;
    double h_start;      /* Log-vol at phase start */
    double h_end;        /* Log-vol at phase end (for gradual transitions) */
    int regime_start;    /* Regime at phase start */
    int regime_end;      /* Regime at phase end */
    double outlier_prob; /* Probability of fat-tail outlier */
    double outlier_mult; /* Multiplier for outlier magnitude */
    int is_gradual;      /* 1 = linear interpolation, 0 = instant at start */
    const char *description;
} TestPhase;

static TestPhase phases[] = {
    /* Phase 0: Burn-in - let filter converge */
    {
        .name = "burn_in",
        .start_tick = 0,
        .end_tick = 500,
        .h_start = -3.5,
        .h_end = -3.5,
        .regime_start = 1,
        .regime_end = 1,
        .outlier_prob = 0.0,
        .outlier_mult = 0.0,
        .is_gradual = 0,
        .description = "Stable burn-in period for filter convergence"},
    /* Phase 1: Outlier test (stable h, outliers present) - should NOT move estimate */
    {
        .name = "outlier_stable",
        .start_tick = 500,
        .end_tick = 1000,
        .h_start = -3.5,
        .h_end = -3.5,
        .regime_start = 1,
        .regime_end = 1,
        .outlier_prob = 0.05,
        .outlier_mult = 8.0,
        .is_gradual = 0,
        .description = "Stable vol with 5% outliers - filter should NOT chase"},
    /* Phase 2: Sudden regime jump (no outliers) - should adapt FAST */
    {
        .name = "sudden_jump",
        .start_tick = 1000,
        .end_tick = 1500,
        .h_start = -2.3,
        .h_end = -2.3,
        .regime_start = 2,
        .regime_end = 2,
        .outlier_prob = 0.0,
        .outlier_mult = 0.0,
        .is_gradual = 0,
        .description = "Sudden vol spike - filter should adapt quickly"},
    /* Phase 3: High vol stable with outliers */
    {
        .name = "high_vol_outliers",
        .start_tick = 1500,
        .end_tick = 2000,
        .h_start = -2.3,
        .h_end = -2.3,
        .regime_start = 2,
        .regime_end = 2,
        .outlier_prob = 0.03,
        .outlier_mult = 6.0,
        .is_gradual = 0,
        .description = "Elevated vol with occasional outliers"},
    /* Phase 4: Gradual decline back to normal */
    {
        .name = "gradual_decline",
        .start_tick = 2000,
        .end_tick = 3000,
        .h_start = -2.3,
        .h_end = -4.0,
        .regime_start = 2,
        .regime_end = 0,
        .outlier_prob = 0.01,
        .outlier_mult = 5.0,
        .is_gradual = 1,
        .description = "Gradual vol decline - filter should follow smoothly"},
    /* Phase 5: Calm stable period */
    {
        .name = "calm_stable",
        .start_tick = 3000,
        .end_tick = 3500,
        .h_start = -4.0,
        .h_end = -4.0,
        .regime_start = 0,
        .regime_end = 0,
        .outlier_prob = 0.0,
        .outlier_mult = 0.0,
        .is_gradual = 0,
        .description = "Very calm market - filter should stay stable"},
    /* Phase 6: Outlier burst (NO real change) - critical test */
    {
        .name = "outlier_burst",
        .start_tick = 3500,
        .end_tick = 3600,
        .h_start = -4.0,
        .h_end = -4.0,
        .regime_start = 0,
        .regime_end = 0,
        .outlier_prob = 0.20,
        .outlier_mult = 10.0,
        .is_gradual = 0,
        .description = "CRITICAL: Outlier burst with NO real change - should NOT adapt"},
    /* Phase 7: Real crash immediately after outlier burst */
    {
        .name = "real_crash",
        .start_tick = 3600,
        .end_tick = 4000,
        .h_start = -2.0,
        .h_end = -2.0,
        .regime_start = 3,
        .regime_end = 3,
        .outlier_prob = 0.08,
        .outlier_mult = 7.0,
        .is_gradual = 0,
        .description = "Real crash after outlier burst - should adapt despite noise"},
    /* Phase 8: Recovery */
    {
        .name = "recovery",
        .start_tick = 4000,
        .end_tick = 5000,
        .h_start = -2.0,
        .h_end = -3.5,
        .regime_start = 3,
        .regime_end = 1,
        .outlier_prob = 0.02,
        .outlier_mult = 5.0,
        .is_gradual = 1,
        .description = "Gradual recovery to normal"}};

#define N_PHASES (sizeof(phases) / sizeof(phases[0]))
#define TOTAL_TICKS 5000

/*============================================================================
 * RNG (same as test_rbpf_scenarios.c)
 *============================================================================*/

static uint64_t rng_state[2];

static inline uint64_t rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoroshiro128plus(void)
{
    const uint64_t s0 = rng_state[0];
    uint64_t s1 = rng_state[1];
    const uint64_t result = s0 + s1;
    s1 ^= s0;
    rng_state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    rng_state[1] = rotl(s1, 37);
    return result;
}

static void rng_seed(uint64_t seed)
{
    rng_state[0] = seed;
    rng_state[1] = seed ^ 0x123456789ABCDEF0ULL;
    for (int i = 0; i < 20; i++)
        xoroshiro128plus();
}

static double rng_uniform(void)
{
    return (xoroshiro128plus() >> 11) * (1.0 / 9007199254740992.0);
}

static double rng_normal(void)
{
    double u1 = rng_uniform();
    double u2 = rng_uniform();
    if (u1 < 1e-10)
        u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

static double rng_student_t(double df)
{
    double n = rng_normal();
    double chi2 = 0.0;
    for (int i = 0; i < (int)df; i++)
    {
        double z = rng_normal();
        chi2 += z * z;
    }
    return n / sqrt(chi2 / df);
}

/*============================================================================
 * SYNTHETIC DATA GENERATION
 *============================================================================*/

typedef struct
{
    int t;
    int phase_id;
    const char *phase_name;
    double y_obs;    /* Observed (transformed): log(r^2) */
    double h_true;   /* True log-volatility */
    double r_true;   /* True return (before transform) */
    int is_outlier;  /* Ground truth: was this an outlier? */
    int regime_true; /* True regime */
} SyntheticTick;

static int get_phase_for_tick(int t)
{
    for (int i = 0; i < (int)N_PHASES; i++)
    {
        if (t >= phases[i].start_tick && t < phases[i].end_tick)
        {
            return i;
        }
    }
    return N_PHASES - 1;
}

static void generate_tick(int t, SyntheticTick *tick, double *h_state)
{
    int phase_id = get_phase_for_tick(t);
    TestPhase *phase = &phases[phase_id];

    tick->t = t;
    tick->phase_id = phase_id;
    tick->phase_name = phase->name;

    /* Compute true h for this tick */
    double h_true;
    int regime_true;

    if (phase->is_gradual && phase->end_tick > phase->start_tick)
    {
        /* Linear interpolation */
        double frac = (double)(t - phase->start_tick) / (phase->end_tick - phase->start_tick);
        h_true = phase->h_start + frac * (phase->h_end - phase->h_start);

        /* Regime also interpolates (discrete) */
        if (frac < 0.5)
        {
            regime_true = phase->regime_start;
        }
        else
        {
            regime_true = phase->regime_end;
        }
    }
    else
    {
        /* Instant change at phase start */
        h_true = phase->h_start;
        regime_true = phase->regime_start;
    }

    /* Add SV dynamics: h_t = mu + phi*(h_{t-1} - mu) + sigma_h * eta */
    double phi = 0.98;
    double sigma_h = 0.15;
    double mu = h_true; /* Use phase target as mean */

    *h_state = mu + phi * (*h_state - mu) + sigma_h * rng_normal();

    /* Clamp h to reasonable range */
    if (*h_state < -6.0)
        *h_state = -6.0;
    if (*h_state > -1.0)
        *h_state = -1.0;

    tick->h_true = *h_state;
    tick->regime_true = regime_true;

    /* Generate return */
    double vol = exp(tick->h_true); /* σ_t = exp(ℓ_t), using log-vol convention */

    /* Check for outlier */
    tick->is_outlier = 0;
    double r;

    if (phase->outlier_prob > 0 && rng_uniform() < phase->outlier_prob)
    {
        /* Generate outlier using Student-t with low df */
        tick->is_outlier = 1;
        r = vol * phase->outlier_mult * rng_student_t(3.0);
    }
    else
    {
        /* Normal return */
        r = vol * rng_normal();
    }

    tick->r_true = r;

    /* Transform to observation: y = log(r^2) */
    double r_sq = r * r;
    if (r_sq < 1e-16)
        r_sq = 1e-16; /* Avoid log(0) */
    tick->y_obs = log(r_sq);
}

/*============================================================================
 * MAIN TEST
 *============================================================================*/

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         RBPF CSV Test - Volatility & Regime Tracking         ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Output: %s                              ║\n", CSV_OUTPUT_DIR);
    printf("║  Ticks: %d | Particles: %d | Regimes: %d                     ║\n",
           TOTAL_TICKS, N_PARTICLES, N_REGIMES);
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* Create output directory */
#ifdef _WIN32
    _mkdir(CSV_OUTPUT_DIR);
#else
    mkdir(CSV_OUTPUT_DIR, 0755);
#endif

    /* Seed RNG */
    rng_seed(RANDOM_SEED);

    /* Create RBPF with Storvik + OCSN */
    printf("  Creating RBPF...\n");
    fflush(stdout);

    RBPF_Extended *ext = rbpf_ext_create(N_PARTICLES, N_REGIMES, RBPF_PARAM_STORVIK);
    if (!ext)
    {
        fprintf(stderr, "Failed to create RBPF\n");
        return 1;
    }
    printf("  RBPF created OK\n");
    fflush(stdout);

    /* Set regime parameters (log-volatility scale)
     * R0: Calm     μ=-4.6 (σ≈1%)
     * R1: Normal   μ=-3.5 (σ≈3%)
     * R2: Elevated μ=-2.5 (σ≈9%)
     * R3: Crisis   μ=-1.5 (σ≈22%)
     */
    printf("  Setting regime params...\n");
    fflush(stdout);
    rbpf_ext_set_regime_params(ext, 0, 0.02f, -4.6f, 0.10f); /* Calm */
    rbpf_ext_set_regime_params(ext, 1, 0.02f, -3.5f, 0.15f); /* Normal */
    rbpf_ext_set_regime_params(ext, 2, 0.02f, -2.5f, 0.20f); /* Elevated */
    rbpf_ext_set_regime_params(ext, 3, 0.02f, -1.5f, 0.25f); /* Crisis */
    printf("  Regime params set OK\n");
    fflush(stdout);

    /* Build transition matrix (sticky regimes) */
    printf("  Building transition LUT...\n");
    fflush(stdout);
    rbpf_real_t trans[16] = {
        0.97f, 0.03f, 0.00f, 0.00f, /* R0 → mostly stay */
        0.02f, 0.95f, 0.03f, 0.00f, /* R1 → can go up/down */
        0.00f, 0.03f, 0.94f, 0.03f, /* R2 → can go up/down */
        0.00f, 0.00f, 0.05f, 0.95f  /* R3 → mostly stay, slow recovery */
    };
    rbpf_ext_build_transition_lut(ext, trans);
    printf("  Transition LUT built OK\n");
    fflush(stdout);

    /* Enable Robust OCSN */
    printf("  Enabling OCSN...\n");
    fflush(stdout);
    rbpf_ext_enable_robust_ocsn(ext);
    printf("  OCSN enabled OK\n");
    fflush(stdout);

    /* Set fixed forgetting per regime */
    param_learn_set_forgetting(&ext->storvik, 1, 0.998f);
    param_learn_set_regime_forgetting(&ext->storvik, 0, 0.999f); /* Calm */
    param_learn_set_regime_forgetting(&ext->storvik, 1, 0.998f); /* Normal */
    param_learn_set_regime_forgetting(&ext->storvik, 2, 0.996f); /* Elevated */
    param_learn_set_regime_forgetting(&ext->storvik, 3, 0.993f); /* Crisis */
    printf("  Forgetting factors set OK\n");
    fflush(stdout);

    /* Initialize filter state */
    printf("  Initializing filter...\n");
    fflush(stdout);
    rbpf_ext_init(ext, -3.5f, 0.5f); /* Start at normal vol with some uncertainty */
    printf("  Filter initialized OK\n");
    fflush(stdout);

    /* Open CSV file */
    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path), "%srbpf_visualization.csv", CSV_OUTPUT_DIR);
    FILE *csv = fopen(csv_path, "w");
    if (!csv)
    {
        fprintf(stderr, "ERROR: Failed to open '%s' for writing\n", csv_path);
        fprintf(stderr, "       Make sure the directory exists: %s\n", CSV_OUTPUT_DIR);
        fprintf(stderr, "       Create it with: mkdir %s\n", CSV_OUTPUT_DIR);
        fflush(stderr);
        rbpf_ext_destroy(ext);
        return 1;
    }
    printf("  Writing to: %s\n\n", csv_path);

    /* Write header - matches rbpf_diagnostic_fixed.ipynb expected format */
    fprintf(csv, "t,true_vol,est_vol_fast,est_vol_smooth,true_log_vol,est_log_vol_fast,est_log_vol_smooth,"
                 "true_regime,est_regime_fast,est_regime_smooth,regime_confidence,surprise,entropy,"
                 "vol_ratio,ess,detected,change_type\n");

    /* Generate and process data */
    double h_state = -3.5; /* Initial h state */
    int current_phase = -1;

    /* Tracking for vol_ratio (EMA-based) */
    double vol_ema_short = 0.0;
    double vol_ema_long = 0.0;
    const double alpha_short = 0.1; /* Fast EMA */
    const double alpha_long = 0.02; /* Slow EMA */
    int prev_regime = -1;

    for (int t = 0; t < TOTAL_TICKS; t++)
    {
        /* Generate synthetic tick */
        SyntheticTick tick;
        generate_tick(t, &tick, &h_state);

        /* Print phase transitions */
        if (tick.phase_id != current_phase)
        {
            current_phase = tick.phase_id;
            printf("  [t=%4d] Phase %d: %s\n", t, current_phase, phases[current_phase].name);
            printf("           %s\n", phases[current_phase].description);
            fflush(stdout);
        }

        /* Run RBPF step */
        RBPF_KSC_Output output;
        memset(&output, 0, sizeof(output));

        if (t == 0)
        {
            printf("  Running first RBPF step...\n");
            printf("    r_true = %.6f, h_true = %.6f\n", tick.r_true, tick.h_true);
            fflush(stdout);
        }

        rbpf_ext_step(ext, (rbpf_real_t)tick.r_true, &output);

        if (t == 0)
        {
            printf("  First step completed\n");
            printf("    h_est=%.3f, ess=%.1f\n", output.log_vol_mean, output.ess);
            fflush(stdout);
        }

        /* Progress indicator */
        if (t > 0 && t % 500 == 0)
        {
            printf("  [t=%4d] Progress: %.0f%% complete\n", t, (100.0 * t) / TOTAL_TICKS);
            fflush(stdout);
        }

        /* Extract estimates */
        double h_est = output.log_vol_mean;
        double vol_est = exp(h_est);
        double vol_true = exp(tick.h_true);
        int regime_est = output.dominant_regime;
        int regime_smooth = output.smoothed_regime;
        double ess = output.ess;
        double surprise = output.surprise;

        /* Compute entropy: -sum(p * log(p)) */
        double entropy = 0.0;
        for (int r = 0; r < N_REGIMES; r++)
        {
            double p = output.regime_probs[r];
            if (p > 1e-10)
            {
                entropy -= p * log(p);
            }
        }

        /* Compute regime confidence (max probability) */
        double confidence = 0.0;
        for (int r = 0; r < N_REGIMES; r++)
        {
            if (output.regime_probs[r] > confidence)
            {
                confidence = output.regime_probs[r];
            }
        }

        /* Update vol EMAs for vol_ratio */
        if (t == 0)
        {
            vol_ema_short = vol_est;
            vol_ema_long = vol_est;
        }
        else
        {
            vol_ema_short = alpha_short * vol_est + (1.0 - alpha_short) * vol_ema_short;
            vol_ema_long = alpha_long * vol_est + (1.0 - alpha_long) * vol_ema_long;
        }
        double vol_ratio = (vol_ema_long > 1e-10) ? vol_ema_short / vol_ema_long : 1.0;

        /* Detect regime change */
        int detected = 0;
        int change_type = 0;
        if (prev_regime >= 0 && regime_est != prev_regime)
        {
            detected = 1;
            change_type = 1; /* Regime shift */
        }
        if (surprise > 5.0)
        {
            detected = 1;
            change_type = (change_type == 1) ? 3 : 2; /* 2=vol_shock, 3=both */
        }
        prev_regime = regime_est;

        /* Smooth estimates (use lag-5 if available, else same as fast) */
        double h_est_smooth = output.smooth_valid ? output.log_vol_mean_smooth : h_est;
        double vol_est_smooth = exp(h_est_smooth);
        int regime_smooth_out = output.smooth_valid ? output.dominant_regime_smooth : regime_smooth;

        /* Write to CSV - format matches notebook expectations */
        fprintf(csv, "%d,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%d,%d,%d,%.3f,%.4f,%.4f,%.4f,%.1f,%d,%d\n",
                t,
                vol_true,          /* true_vol */
                vol_est,           /* est_vol_fast */
                vol_est_smooth,    /* est_vol_smooth */
                tick.h_true,       /* true_log_vol */
                h_est,             /* est_log_vol_fast */
                h_est_smooth,      /* est_log_vol_smooth */
                tick.regime_true,  /* true_regime */
                regime_est,        /* est_regime_fast */
                regime_smooth_out, /* est_regime_smooth */
                confidence,        /* regime_confidence */
                surprise,          /* surprise */
                entropy,           /* entropy */
                vol_ratio,         /* vol_ratio */
                ess,               /* ess */
                detected,          /* detected */
                change_type);      /* change_type */
    }

    fclose(csv);
    printf("\n✓ Wrote %s\n", csv_path);

    /* Write phase summary */
    snprintf(csv_path, sizeof(csv_path), "%sphases.csv", CSV_OUTPUT_DIR);
    csv = fopen(csv_path, "w");
    if (csv)
    {
        fprintf(csv, "phase_id,name,start_tick,end_tick,h_start,h_end,regime_start,regime_end,"
                     "outlier_prob,is_gradual,description\n");
        for (int i = 0; i < (int)N_PHASES; i++)
        {
            fprintf(csv, "%d,%s,%d,%d,%.2f,%.2f,%d,%d,%.2f,%d,\"%s\"\n",
                    i, phases[i].name, phases[i].start_tick, phases[i].end_tick,
                    phases[i].h_start, phases[i].h_end,
                    phases[i].regime_start, phases[i].regime_end,
                    phases[i].outlier_prob, phases[i].is_gradual,
                    phases[i].description);
        }
        fclose(csv);
        printf("✓ Wrote %s\n", csv_path);
    }

    /* Cleanup */
    rbpf_ext_destroy(ext);

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Done! Open the Jupyter notebook to visualize results.\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    return 0;
}