/**
 * @file test_rbpf_scenarios.c
 * @brief Realistic scenario tests for RBPF Pipeline (SSA → RBPF → Kelly stack)
 *
 * Tests RBPF's unified change detection + volatility tracking against:
 *   1. Flash crash (sudden vol spike, quick recovery)
 *   2. Fed announcement (scheduled event, vol crush after)
 *   3. Earnings gap (overnight gap, elevated vol)
 *   4. Liquidity crisis (persistent high vol, negative drift)
 *   5. Gradual regime shift (slow transition)
 *   6. Overnight gap (gap opening from news)
 *   7. Intraday pattern (lunch lull → power hour)
 *   8. Correlation spike (stress event)
 *   9. Oscillating regimes (low→high→low vol)
 *  10. Pre-crisis buildup (trending vol / VIX creep)
 *
 * Validates:
 *   - Change detection (surprise, vol_ratio signals)
 *   - Regime identification accuracy
 *   - Volatility tracking (MAE vs true vol)
 *   - Position scaling behavior
 *   - Detection delay distribution
 *
 * Compile:
 *   icx -O3 -xHost -qopenmp test_rbpf_scenarios.c rbpf_ksc.c rbpf_pipeline.c \
 *       -o test_scenarios -qmkl -DNDEBUG
 */

#include "rbpf_ksc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

/*============================================================================
 * CONFIGURATION
 *============================================================================*/

#ifndef N_MONTE_CARLO
#define N_MONTE_CARLO 30 /* Simulations per scenario */
#endif

#ifndef N_OBSERVATIONS
#define N_OBSERVATIONS 1000 /* Ticks per simulation */
#endif

#ifndef N_PARTICLES
#define N_PARTICLES 200 /* RBPF particles */
#endif

/*============================================================================
 * RNG (xorshift128+)
 *============================================================================*/

static uint64_t rng_s[2];

static void rng_seed(uint64_t seed)
{
    rng_s[0] = seed;
    rng_s[1] = seed ^ 0xDEADBEEFCAFEBABEULL;
    for (int i = 0; i < 20; i++)
    {
        rng_s[0] ^= rng_s[0] << 13;
        rng_s[0] ^= rng_s[0] >> 7;
        rng_s[0] ^= rng_s[0] << 17;
    }
}

static inline uint64_t rng_next(void)
{
    uint64_t s1 = rng_s[0];
    const uint64_t s0 = rng_s[1];
    rng_s[0] = s0;
    s1 ^= s1 << 23;
    rng_s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return rng_s[1] + s0;
}

static inline double randu(void)
{
    return (rng_next() >> 11) * (1.0 / 9007199254740992.0);
}

static inline double randn(void)
{
    double u1 = randu(), u2 = randu();
    if (u1 < 1e-10)
        u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

static double rand_student_t(double df)
{
    double z = randn();
    double chi2 = 0.0;
    for (int i = 0; i < (int)df; i++)
    {
        double g = randn();
        chi2 += g * g;
    }
    return z / sqrt(chi2 / df);
}

/*============================================================================
 * STOCHASTIC VOLATILITY MODEL
 *============================================================================*/

typedef struct
{
    double drift;
    double mu_vol;
    double theta_vol;
    double sigma_vol;
    double rho;
    double jump_intensity;
    double jump_mean;
    double jump_std;
    double student_df;
} SVParams;

static SVParams sv_normal_market(void)
{
    return (SVParams){
        .drift = 0.0,
        .mu_vol = -4.0,
        .theta_vol = 0.02,
        .sigma_vol = 0.05,
        .rho = -0.5,
        .jump_intensity = 0.0,
        .jump_mean = 0.0,
        .jump_std = 0.0,
        .student_df = 0};
}

static SVParams sv_crisis(void)
{
    /* Crisis scenario: high vol with jumps
     *
     * NOTE: Using student_df=5 for realistic fat tails.
     * The Omori mixture (10-component) is specifically designed to handle
     * fat-tailed returns - it needs them to trigger high-likelihood tail
     * components properly. Gaussian returns underutilize the model.
     */
    return (SVParams){
        .drift = -0.001,
        .mu_vol = -2.5,
        .theta_vol = 0.01,
        .sigma_vol = 0.15,
        .rho = -0.7,
        .jump_intensity = 0.02,
        .jump_mean = 0.3,
        .jump_std = 0.2,
        .student_df = 5 /* Fat tails - Omori mixture handles these */
    };
}

static void sv_step(double *log_vol, double *price, const SVParams *p, double *ret_out)
{
    double z1 = randn();
    double z2 = p->rho * z1 + sqrt(1.0 - p->rho * p->rho) * randn();

    if (p->student_df > 0)
    {
        z1 = rand_student_t(p->student_df) / sqrt(p->student_df / (p->student_df - 2));
    }

    double new_log_vol = (1.0 - p->theta_vol) * (*log_vol) +
                         p->theta_vol * p->mu_vol +
                         p->sigma_vol * z2;

    if (p->jump_intensity > 0 && randu() < p->jump_intensity)
    {
        new_log_vol += p->jump_mean + p->jump_std * randn();
    }

    *log_vol = new_log_vol;
    double vol = exp(*log_vol);
    double ret = p->drift + vol * z1;
    *price = (*price) * (1.0 + ret);
    *ret_out = ret;
}

/*============================================================================
 * SCENARIO DEFINITIONS
 *============================================================================*/

typedef struct
{
    const char *name;
    const char *description;
    int changepoint;
    int transition_ticks;
    SVParams before;
    SVParams after;
    int has_second_change;
    int second_changepoint;
    SVParams final;

    /* Expected RBPF regime mapping */
    int expected_regime_before; /* 0=calm, 1=normal, 2=elevated, 3=crisis */
    int expected_regime_after;
} Scenario;

/* Scenario 1: Flash Crash */
static Scenario scenario_flash_crash(void)
{
    SVParams normal = sv_normal_market();
    SVParams crash = sv_crisis();
    crash.drift = -0.003;
    crash.mu_vol = -2.0;
    crash.jump_intensity = 0.05;

    SVParams recovery = sv_normal_market();
    recovery.mu_vol = -3.5;

    return (Scenario){
        .name = "Flash Crash",
        .description = "Sudden vol spike with quick recovery",
        .changepoint = 400,
        .transition_ticks = 0,
        .before = normal,
        .after = crash,
        .has_second_change = 1,
        .second_changepoint = 450,
        .final = recovery,
        .expected_regime_before = 0,
        .expected_regime_after = 2 /* mu_vol=-2.0 < -1.85 → regime 2 */
    };
}

/* Scenario 2: Fed Announcement */
static Scenario scenario_fed_announcement(void)
{
    SVParams pre_fed = sv_normal_market();
    pre_fed.sigma_vol = 0.08;

    SVParams post_fed = sv_normal_market();
    post_fed.mu_vol = -3.0;
    post_fed.drift = 0.0005;

    SVParams crush = sv_normal_market();
    crush.mu_vol = -4.5;
    crush.sigma_vol = 0.03;

    return (Scenario){
        .name = "Fed Announcement",
        .description = "Scheduled event with vol spike then crush",
        .changepoint = 500,
        .transition_ticks = 0,
        .before = pre_fed,
        .after = post_fed,
        .has_second_change = 1,
        .second_changepoint = 550,
        .final = crush,
        .expected_regime_before = 0,
        .expected_regime_after = 2};
}

/* Scenario 3: Earnings Surprise */
static Scenario scenario_earnings_gap(void)
{
    SVParams normal = sv_normal_market();
    SVParams gap = sv_normal_market();
    gap.drift = 0.002;
    gap.mu_vol = -3.0;
    gap.sigma_vol = 0.10;
    gap.jump_intensity = 0.01;

    return (Scenario){
        .name = "Earnings Surprise",
        .description = "Gap up with elevated post-earnings vol",
        .changepoint = 450,
        .transition_ticks = 5,
        .before = normal,
        .after = gap,
        .has_second_change = 0,
        .expected_regime_before = 0,
        .expected_regime_after = 2};
}

/* Scenario 4: Liquidity Crisis */
static Scenario scenario_liquidity_crisis(void)
{
    SVParams normal = sv_normal_market();
    SVParams crisis = sv_crisis();
    crisis.theta_vol = 0.005;
    crisis.rho = -0.8;

    return (Scenario){
        .name = "Liquidity Crisis",
        .description = "Persistent high vol, slow recovery",
        .changepoint = 400,
        .transition_ticks = 50,
        .before = normal,
        .after = crisis,
        .has_second_change = 0,
        .expected_regime_before = 0,
        .expected_regime_after = 2 /* mu_vol=-2.5 → regime 2 */
    };
}

/* Scenario 5: Gradual Trend Change */
static Scenario scenario_gradual_shift(void)
{
    SVParams bull = sv_normal_market();
    bull.drift = 0.0005;
    bull.mu_vol = -4.2;

    SVParams bear = sv_normal_market();
    bear.drift = -0.0003;
    bear.mu_vol = -3.5;
    bear.sigma_vol = 0.08;

    return (Scenario){
        .name = "Gradual Regime Shift",
        .description = "Slow transition from bull to bear",
        .changepoint = 300,
        .transition_ticks = 200,
        .before = bull,
        .after = bear,
        .has_second_change = 0,
        .expected_regime_before = 0,
        .expected_regime_after = 1};
}

/* Scenario 6: Overnight Gap */
static Scenario scenario_overnight_gap(void)
{
    SVParams normal = sv_normal_market();
    SVParams gap = sv_normal_market();
    gap.drift = 0.003;
    gap.mu_vol = -2.8;
    gap.sigma_vol = 0.12;
    gap.jump_intensity = 0.02;

    SVParams post_gap = sv_normal_market();
    post_gap.mu_vol = -3.5;
    post_gap.theta_vol = 0.03;

    return (Scenario){
        .name = "Overnight Gap",
        .description = "Gap opening then normalization",
        .changepoint = 400,
        .transition_ticks = 0,
        .before = normal,
        .after = gap,
        .has_second_change = 1,
        .second_changepoint = 420,
        .final = post_gap,
        .expected_regime_before = 0,
        .expected_regime_after = 2};
}

/* Scenario 7: Intraday Pattern */
static Scenario scenario_intraday_pattern(void)
{
    SVParams morning = sv_normal_market();
    morning.mu_vol = -3.8;

    SVParams lunch = sv_normal_market();
    lunch.mu_vol = -4.8;
    lunch.sigma_vol = 0.03;

    SVParams power_hour = sv_normal_market();
    power_hour.mu_vol = -3.2;
    power_hour.sigma_vol = 0.10;

    return (Scenario){
        .name = "Intraday Pattern",
        .description = "Lunch lull then power hour",
        .changepoint = 350,
        .transition_ticks = 20,
        .before = morning,
        .after = lunch,
        .has_second_change = 1,
        .second_changepoint = 600,
        .final = power_hour,
        .expected_regime_before = 1,
        .expected_regime_after = 0};
}

/* Scenario 8: Correlation Spike */
static Scenario scenario_correlation_spike(void)
{
    SVParams normal = sv_normal_market();
    normal.rho = -0.3;

    SVParams stress = sv_crisis();
    stress.rho = -0.8;
    stress.mu_vol = -2.5;
    stress.sigma_vol = 0.15;
    stress.jump_intensity = 0.03;

    return (Scenario){
        .name = "Correlation Spike",
        .description = "Stress event with correlation breakdown",
        .changepoint = 450,
        .transition_ticks = 10,
        .before = normal,
        .after = stress,
        .has_second_change = 0,
        .expected_regime_before = 0,
        .expected_regime_after = 2 /* mu_vol=-2.5 → regime 2 */
    };
}

/* Scenario 9: Oscillating Regimes */
static Scenario scenario_oscillating(void)
{
    SVParams low_vol = sv_normal_market();
    low_vol.mu_vol = -4.5;
    low_vol.sigma_vol = 0.04;

    SVParams high_vol = sv_normal_market();
    high_vol.mu_vol = -3.0;
    high_vol.sigma_vol = 0.12;

    return (Scenario){
        .name = "Oscillating Regimes",
        .description = "Low→High→Low vol oscillation",
        .changepoint = 300,
        .transition_ticks = 0,
        .before = low_vol,
        .after = high_vol,
        .has_second_change = 1,
        .second_changepoint = 550,
        .final = low_vol,
        .expected_regime_before = 0,
        .expected_regime_after = 2};
}

/* Scenario 10: Pre-Crisis Buildup */
static Scenario scenario_trending_vol(void)
{
    SVParams calm = sv_normal_market();
    calm.mu_vol = -4.5;
    calm.sigma_vol = 0.03;

    SVParams building = sv_normal_market();
    building.mu_vol = -3.0;
    building.sigma_vol = 0.10;
    building.theta_vol = 0.005;
    building.drift = -0.0001;

    return (Scenario){
        .name = "Pre-Crisis Buildup",
        .description = "Gradually increasing vol (VIX creep)",
        .changepoint = 250,
        .transition_ticks = 400,
        .before = calm,
        .after = building,
        .has_second_change = 0,
        .expected_regime_before = 0,
        .expected_regime_after = 2};
}

/*============================================================================
 * DATA GENERATION
 *============================================================================*/

static SVParams sv_interpolate(const SVParams *a, const SVParams *b, double t)
{
    SVParams p;
    p.drift = a->drift + t * (b->drift - a->drift);
    p.mu_vol = a->mu_vol + t * (b->mu_vol - a->mu_vol);
    p.theta_vol = a->theta_vol + t * (b->theta_vol - a->theta_vol);
    p.sigma_vol = a->sigma_vol + t * (b->sigma_vol - a->sigma_vol);
    p.rho = a->rho + t * (b->rho - a->rho);
    p.jump_intensity = a->jump_intensity + t * (b->jump_intensity - a->jump_intensity);
    p.jump_mean = a->jump_mean + t * (b->jump_mean - a->jump_mean);
    p.jump_std = a->jump_std + t * (b->jump_std - a->jump_std);
    p.student_df = a->student_df + t * (b->student_df - a->student_df);
    return p;
}

typedef struct
{
    double *returns;
    double *true_vol;
    double *true_log_vol;
    int *true_regime; /* Simplified: 0=calm, 1=normal, 2=elevated, 3=crisis */
    int n;
} ScenarioData;

static int classify_vol_to_regime(double log_vol)
{
    /* Map log-vol to regime (aligned with RBPF mu_vol centers):
     *   Regime 0: μ=-4.6, boundary <= -4.05 → 0 (calm, ~1.0%)
     *   Regime 1: μ=-3.5, -4.05 to -3.0     → 1 (normal, ~3%)
     *   Regime 2: μ=-2.5, -3.0 to -1.85     → 2 (elevated, ~8%)
     *   Regime 3: μ=-1.2, > -1.85           → 3 (crisis, ~30%)
     *
     * Boundaries at midpoints between regime centers.
     */
    if (log_vol <= -4.05)
        return 0;
    if (log_vol < -3.0)
        return 1;
    if (log_vol < -1.85)
        return 2;
    return 3;
}

static ScenarioData *generate_scenario_data(const Scenario *s, int n)
{
    ScenarioData *data = (ScenarioData *)malloc(sizeof(ScenarioData));
    data->n = n;
    data->returns = (double *)malloc(n * sizeof(double));
    data->true_vol = (double *)malloc(n * sizeof(double));
    data->true_log_vol = (double *)malloc(n * sizeof(double));
    data->true_regime = (int *)malloc(n * sizeof(int));

    double price = 100.0;
    double log_vol = s->before.mu_vol;

    for (int t = 0; t < n; t++)
    {
        SVParams params;

        if (t < s->changepoint)
        {
            params = s->before;
        }
        else if (s->transition_ticks > 0 && t < s->changepoint + s->transition_ticks)
        {
            double frac = (double)(t - s->changepoint) / s->transition_ticks;
            params = sv_interpolate(&s->before, &s->after, frac);
        }
        else if (s->has_second_change && t >= s->second_changepoint)
        {
            params = s->final;
        }
        else
        {
            params = s->after;
        }

        double ret;
        sv_step(&log_vol, &price, &params, &ret);

        data->returns[t] = ret;
        data->true_vol[t] = exp(log_vol);
        data->true_log_vol[t] = log_vol;
        data->true_regime[t] = classify_vol_to_regime(log_vol);
    }

    return data;
}

static void free_scenario_data(ScenarioData *data)
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

/*============================================================================
 * RBPF PIPELINE EVALUATION
 *============================================================================*/

typedef struct
{
    /* Change detection */
    double mean_detection_delay;
    double std_detection_delay;
    double detection_rate; /* % of runs where change was detected */
    double false_positive_rate;

    /* Regime tracking */
    double regime_accuracy;      /* % correct regime classification */
    double regime_accuracy_post; /* Accuracy after changepoint */

    /* Volatility tracking */
    double vol_mae;         /* Mean absolute error on vol */
    double vol_mae_post;    /* MAE after changepoint */
    double vol_correlation; /* Correlation with true vol */

    /* Position scaling behavior */
    double avg_scale_before;    /* Average position_scale before change */
    double avg_scale_during;    /* Average during transition */
    double avg_scale_after;     /* Average after settling */
    double min_scale_at_change; /* Minimum scale around changepoint */

    /* Timing */
    double avg_latency_us;

    int n_runs;
} ScenarioStats;

static void run_scenario_evaluation(const Scenario *s, int n_runs, ScenarioStats *stats)
{
    memset(stats, 0, sizeof(ScenarioStats));
    stats->n_runs = n_runs;

    double sum_delay = 0, sum_delay2 = 0;
    int n_detected = 0, n_false_pos = 0;
    double sum_regime_acc = 0, sum_regime_acc_post = 0;
    double sum_vol_mae = 0, sum_vol_mae_post = 0;
    int n_valid_vol_runs = 0, n_valid_vol_runs_post = 0;
    double sum_scale_before = 0, sum_scale_during = 0, sum_scale_after = 0;
    double sum_min_scale = 0;
    double sum_latency = 0;
    int total_ticks = 0;

    int warmup = 150;
    int detection_window = 100; /* Look for detection within this window after CP */

    for (int run = 0; run < n_runs; run++)
    {
        rng_seed(12345 + run * 7919);

        ScenarioData *data = generate_scenario_data(s, N_OBSERVATIONS);

        /* Create RBPF pipeline */
        RBPF_PipelineConfig cfg = rbpf_pipeline_default_config();
        cfg.n_particles = N_PARTICLES;
        RBPF_Pipeline *pipe = rbpf_pipeline_create(&cfg);
        rbpf_pipeline_init(pipe, 0.01f);

        /* Storage for this run */
        int detection_time = -1;
        int had_false_positive = 0; /* Binary: did we have an early false alarm? */
        int regime_correct = 0, regime_correct_post = 0;
        int regime_count = 0, regime_count_post = 0;
        double vol_err_sum = 0, vol_err_sum_post = 0;
        int vol_count = 0, vol_count_post = 0;
        double scale_before_sum = 0, scale_during_sum = 0, scale_after_sum = 0;
        int count_before = 0, count_during = 0, count_after = 0;
        double min_scale = 1.0;

        /* Process all ticks */
        for (int t = 0; t < data->n; t++)
        {
            RBPF_Signal sig;

            double t0 = 0, t1 = 0;
#ifdef _WIN32
            LARGE_INTEGER freq, start, end;
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
#else
            struct timespec ts0, ts1;
            clock_gettime(CLOCK_MONOTONIC, &ts0);
#endif

            rbpf_pipeline_step(pipe, (rbpf_real_t)data->returns[t], &sig);

#ifdef _WIN32
            QueryPerformanceCounter(&end);
            sum_latency += (double)(end.QuadPart - start.QuadPart) / freq.QuadPart * 1e6;
#else
            clock_gettime(CLOCK_MONOTONIC, &ts1);
            sum_latency += (ts1.tv_sec - ts0.tv_sec) * 1e6 + (ts1.tv_nsec - ts0.tv_nsec) / 1e3;
#endif
            total_ticks++;

            /* Skip warmup for metrics */
            if (t < warmup)
                continue;

            /* Change detection */
            if (detection_time < 0 && sig.change_detected >= 2)
            {
                /* Major change detected */
                if (t >= s->changepoint - 20 && t <= s->changepoint + detection_window)
                {
                    detection_time = t;
                }
                else if (t < s->changepoint - 20)
                {
                    had_false_positive = 1; /* Mark this run as having FP */
                }
            }

            /* Regime accuracy */
            if (sig.regime == data->true_regime[t])
            {
                regime_correct++;
            }
            regime_count++;

            if (t > s->changepoint + 50)
            {
                if (sig.regime == data->true_regime[t])
                {
                    regime_correct_post++;
                }
                regime_count_post++;
            }

            /* Vol tracking - guard against NaN/Inf */
            double vol_err = fabs((double)sig.vol_forecast - data->true_vol[t]);
            if (isfinite(vol_err) && isfinite(data->true_vol[t]))
            {
                vol_err_sum += vol_err;
                vol_count++;

                if (t > s->changepoint + 50)
                {
                    vol_err_sum_post += vol_err;
                    vol_count_post++;
                }
            }

            /* Position scaling */
            if (t < s->changepoint - 50)
            {
                scale_before_sum += sig.position_scale;
                count_before++;
            }
            else if (t >= s->changepoint - 10 && t <= s->changepoint + 50)
            {
                scale_during_sum += sig.position_scale;
                count_during++;
                if (sig.position_scale < min_scale)
                {
                    min_scale = sig.position_scale;
                }
            }
            else if (t > s->changepoint + 100)
            {
                scale_after_sum += sig.position_scale;
                count_after++;
            }
        }

        /* Aggregate run results */
        if (detection_time >= 0)
        {
            int delay = detection_time - s->changepoint;
            sum_delay += delay;
            sum_delay2 += delay * delay;
            n_detected++;
        }

        if (had_false_positive)
        {
            n_false_pos++;
        }

        if (regime_count > 0)
        {
            sum_regime_acc += (double)regime_correct / regime_count;
        }
        if (regime_count_post > 0)
        {
            sum_regime_acc_post += (double)regime_correct_post / regime_count_post;
        }
        if (vol_count > 0)
        {
            sum_vol_mae += vol_err_sum / vol_count;
            n_valid_vol_runs++;
        }
        if (vol_count_post > 0)
        {
            sum_vol_mae_post += vol_err_sum_post / vol_count_post;
            n_valid_vol_runs_post++;
        }

        if (count_before > 0)
            sum_scale_before += scale_before_sum / count_before;
        if (count_during > 0)
            sum_scale_during += scale_during_sum / count_during;
        if (count_after > 0)
            sum_scale_after += scale_after_sum / count_after;
        sum_min_scale += min_scale;

        rbpf_pipeline_destroy(pipe);
        free_scenario_data(data);
    }

    /* Compute final statistics */
    if (n_detected > 0)
    {
        stats->mean_detection_delay = sum_delay / n_detected;
        stats->std_detection_delay = sqrt(sum_delay2 / n_detected -
                                          stats->mean_detection_delay * stats->mean_detection_delay);
    }
    stats->detection_rate = (double)n_detected / n_runs * 100.0;
    stats->false_positive_rate = (double)n_false_pos / n_runs * 100.0;

    stats->regime_accuracy = sum_regime_acc / n_runs * 100.0;
    stats->regime_accuracy_post = sum_regime_acc_post / n_runs * 100.0;

    stats->vol_mae = n_valid_vol_runs > 0 ? sum_vol_mae / n_valid_vol_runs : 0.0;
    stats->vol_mae_post = n_valid_vol_runs_post > 0 ? sum_vol_mae_post / n_valid_vol_runs_post : 0.0;

    stats->avg_scale_before = sum_scale_before / n_runs;
    stats->avg_scale_during = sum_scale_during / n_runs;
    stats->avg_scale_after = sum_scale_after / n_runs;
    stats->min_scale_at_change = sum_min_scale / n_runs;

    stats->avg_latency_us = sum_latency / total_ticks;
}

/*============================================================================
 * REPORTING
 *============================================================================*/

static void print_scenario_results(const Scenario *s, const ScenarioStats *stats)
{
    printf("\n┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ %-67s │\n", s->name);
    printf("├─────────────────────────────────────────────────────────────────────┤\n");
    printf("│ %-67s │\n", s->description);
    printf("├─────────────────────────────────────────────────────────────────────┤\n");
    printf("│ Changepoint: t=%d", s->changepoint);
    if (s->transition_ticks > 0)
        printf(" (gradual, %d ticks)", s->transition_ticks);
    printf("%*s│\n", s->transition_ticks > 0 ? 30 : 47, "");
    printf("│ Before: μ_v=%.2f → After: μ_v=%.2f (regime %d→%d)                   │\n",
           s->before.mu_vol, s->after.mu_vol,
           s->expected_regime_before, s->expected_regime_after);
    printf("├─────────────────────────────────────────────────────────────────────┤\n");
    printf("│ CHANGE DETECTION (n=%d runs)                                        │\n", stats->n_runs);
    printf("│   Detection rate:   %5.1f%%                                         │\n", stats->detection_rate);
    printf("│   Mean delay:       %+5.1f ticks (std=%.1f)                          │\n",
           stats->mean_detection_delay, stats->std_detection_delay);
    printf("│   False positive:   %5.1f%%                                         │\n", stats->false_positive_rate);
    printf("├─────────────────────────────────────────────────────────────────────┤\n");
    printf("│ REGIME TRACKING                                                     │\n");
    printf("│   Overall accuracy: %5.1f%%                                         │\n", stats->regime_accuracy);
    printf("│   Post-change acc:  %5.1f%%                                         │\n", stats->regime_accuracy_post);
    printf("├─────────────────────────────────────────────────────────────────────┤\n");
    printf("│ VOLATILITY TRACKING                                                 │\n");
    printf("│   MAE (overall):    %.4f                                           │\n", stats->vol_mae);
    printf("│   MAE (post-change):%.4f                                           │\n", stats->vol_mae_post);
    printf("├─────────────────────────────────────────────────────────────────────┤\n");
    printf("│ POSITION SCALING (Kelly multiplier)                                 │\n");
    printf("│   Before change:    %.2f                                            │\n", stats->avg_scale_before);
    printf("│   During change:    %.2f                                            │\n", stats->avg_scale_during);
    printf("│   After settling:   %.2f                                            │\n", stats->avg_scale_after);
    printf("│   Min at change:    %.2f                                            │\n", stats->min_scale_at_change);
    printf("├─────────────────────────────────────────────────────────────────────┤\n");
    printf("│ LATENCY: %.2f μs/tick                                               │\n", stats->avg_latency_us);
    printf("└─────────────────────────────────────────────────────────────────────┘\n");
}

static void print_summary_table(Scenario *scenarios, ScenarioStats *stats, int n)
{
    printf("\n╔═══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                              SUMMARY                                          ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Scenario             │ Detect │ Delay │ Regime │ Vol MAE │ Scale@CP │ Latency ║\n");
    printf("║                      │   %%    │ ticks │   %%    │         │   min    │   μs    ║\n");
    printf("╠══════════════════════╪════════╪═══════╪════════╪═════════╪══════════╪═════════╣\n");

    for (int i = 0; i < n; i++)
    {
        printf("║ %-20s │ %5.1f%% │ %+5.0f │ %5.1f%% │ %.4f  │   %.2f   │ %6.2f  ║\n",
               scenarios[i].name,
               stats[i].detection_rate,
               stats[i].mean_detection_delay,
               stats[i].regime_accuracy_post,
               stats[i].vol_mae,
               stats[i].min_scale_at_change,
               stats[i].avg_latency_us);
    }

    printf("╚══════════════════════╧════════╧═══════╧════════╧═════════╧══════════╧═════════╝\n");
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(void)
{
    printf("╔═══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║              RBPF PIPELINE: REALISTIC SCENARIO TESTS                          ║\n");
    printf("║                                                                               ║\n");
    printf("║  Stack: SSA → RBPF (change detection + vol tracking) → Kelly                  ║\n");
    printf("║  Monte Carlo: %d runs × %d ticks, Particles: %d                              ║\n",
           N_MONTE_CARLO, N_OBSERVATIONS, N_PARTICLES);
    printf("╚═══════════════════════════════════════════════════════════════════════════════╝\n");

    Scenario scenarios[] = {
        scenario_flash_crash(),
        scenario_fed_announcement(),
        scenario_earnings_gap(),
        scenario_liquidity_crisis(),
        scenario_gradual_shift(),
        scenario_overnight_gap(),
        scenario_intraday_pattern(),
        scenario_correlation_spike(),
        scenario_oscillating(),
        scenario_trending_vol()};
    int n_scenarios = sizeof(scenarios) / sizeof(scenarios[0]);

    ScenarioStats all_stats[10];

    for (int i = 0; i < n_scenarios; i++)
    {
        printf("\nRunning: %s", scenarios[i].name);
        fflush(stdout);

        run_scenario_evaluation(&scenarios[i], N_MONTE_CARLO, &all_stats[i]);

        printf(" ✓\n");
        print_scenario_results(&scenarios[i], &all_stats[i]);
    }

    print_summary_table(scenarios, all_stats, n_scenarios);

    /* Overall assessment */
    printf("\n═══════════════════════════════════════════════════════════════════════════════\n");
    printf("ASSESSMENT:\n\n");

    double avg_detection = 0, avg_regime = 0, avg_vol_mae = 0, avg_scale_drop = 0;
    double crisis_detection = 0;
    int n_crisis = 0;

    for (int i = 0; i < n_scenarios; i++)
    {
        avg_detection += all_stats[i].detection_rate;
        avg_regime += all_stats[i].regime_accuracy_post;
        avg_vol_mae += all_stats[i].vol_mae;
        avg_scale_drop += (all_stats[i].avg_scale_before - all_stats[i].min_scale_at_change);

        /* Track crisis scenarios separately (Liquidity Crisis, Correlation Spike) */
        if (i == 3 || i == 7)
        { /* indices for crisis scenarios */
            crisis_detection += all_stats[i].detection_rate;
            n_crisis++;
        }
    }
    avg_detection /= n_scenarios;
    avg_regime /= n_scenarios;
    avg_vol_mae /= n_scenarios;
    avg_scale_drop /= n_scenarios;
    crisis_detection /= n_crisis;

    printf("  Average detection rate:     %.1f%%\n", avg_detection);
    printf("  Crisis detection rate:      %.1f%%\n", crisis_detection);
    printf("  Average regime accuracy:    %.1f%%\n", avg_regime);
    printf("  Average vol MAE:            %.4f\n", avg_vol_mae);
    printf("  Average scale drop at CP:   %.2f\n", avg_scale_drop);
    printf("\n");

    /* Assessment criteria for Kelly sizing:
     * 1. Crisis detection must be near 100% (these are the dangerous events)
     * 2. Vol tracking must be good (MAE < 0.01)
     * 3. Position scaling must work (scale drops meaningfully at changes)
     * Regime accuracy is less critical - we use vol_forecast directly for Kelly
     */
    int crisis_ok = (crisis_detection >= 95.0);
    int vol_ok = (avg_vol_mae < 0.010);
    int scale_ok = (avg_scale_drop > 0.40);

    if (crisis_ok && vol_ok && scale_ok)
    {
        printf("  ✓ RBPF pipeline ready for real data testing\n");
        printf("    - Crisis detection: %.0f%% (required: ≥95%%)\n", crisis_detection);
        printf("    - Vol MAE: %.4f (required: <0.01)\n", avg_vol_mae);
        printf("    - Scale drop: %.2f (required: >0.40)\n", avg_scale_drop);
    }
    else
    {
        printf("  ✗ Issues found:\n");
        if (!crisis_ok)
            printf("    - Crisis detection %.0f%% < 95%%\n", crisis_detection);
        if (!vol_ok)
            printf("    - Vol MAE %.4f >= 0.01\n", avg_vol_mae);
        if (!scale_ok)
            printf("    - Scale drop %.2f <= 0.40\n", avg_scale_drop);
    }

    printf("═══════════════════════════════════════════════════════════════════════════════\n");

    return 0;
}