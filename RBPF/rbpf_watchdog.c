/**
 * @file rbpf_watchdog.c
 * @brief Divergence Watchdog ("Defibrillator") Implementation
 *
 * This module monitors RBPF health and triggers controlled re-initialization
 * when the particle filter diverges. It's the critical safety rail for
 * production deployment.
 *
 * PHILOSOPHY:
 *   - Fail gracefully, never crash
 *   - Preserve as much state as possible
 *   - Log everything for post-mortem
 *   - Signal upstream systems
 */

#include "rbpf_watchdog.h"
#include "rbpf_ksc_param_integration.h" /* For RBPF_Extended */
#include "rbpf_param_learn.h"           /* For param_learn_broadcast_priors */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

/*═══════════════════════════════════════════════════════════════════════════
 * STRING TABLES
 *═══════════════════════════════════════════════════════════════════════════*/

static const char *REASON_STRINGS[] = {
    "NONE",
    "ESS_COLLAPSE",
    "NAN_WEIGHT",
    "INF_WEIGHT",
    "ZERO_WEIGHTS",
    "LIK_COLLAPSE",
    "LIK_NAN",
    "STATE_DIVERGENCE",
    "STATE_NAN",
    "STATE_EXPLOSION",
    "REGIME_DEADLOCK",
    "MANUAL"};

static const char *SEVERITY_STRINGS[] = {
    "NONE",
    "SOFT",
    "MEDIUM",
    "HARD",
    "CRITICAL"};

/*═══════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_watchdog_default_config(RBPF_WatchdogConfig *config)
{
    if (!config)
        return;

    config->enabled = 1;

    /* ESS thresholds */
    config->ess_critical_ratio = 0.05f; /* 5% of N particles */
    config->ess_warning_ratio = 0.15f;  /* 15% of N particles */

    /* Likelihood thresholds */
    config->log_lik_floor = -1e6f;
    config->log_lik_warning = -1e4f;

    /* State divergence */
    config->state_divergence_sigma = 8.0f; /* 8 sigma is extreme */
    config->state_max_abs = 10.0f;         /* |h| > 10 is ~22000% vol */

    /* Regime deadlock */
    config->regime_deadlock_window = 10;
    config->regime_deadlock_thresh = 0.05f; /* 5% avg return while in R0 */

    /* Cooldown */
    config->min_ticks_between_resets = 50;
    config->max_resets_per_session = 10; /* Circuit breaker */

    /* Reset behavior */
    config->preserve_learned_params = 1; /* Keep Storvik stats */
    config->reset_variance_scale = 2.0f; /* Double init variance */
    config->clear_hawkes_on_reset = 1;   /* Fresh start for Hawkes */
}

void rbpf_watchdog_init(RBPF_Watchdog *wd)
{
    if (!wd)
        return;

    memset(wd, 0, sizeof(RBPF_Watchdog));
    rbpf_watchdog_default_config(&wd->config);

    wd->ticks_since_last_reset = wd->config.min_ticks_between_resets; /* Allow immediate first reset */
    wd->last_reset_reason = RESET_NONE;
    wd->last_reset_severity = SEVERITY_NONE;
}

void rbpf_watchdog_init_config(RBPF_Watchdog *wd, const RBPF_WatchdogConfig *config)
{
    if (!wd)
        return;

    memset(wd, 0, sizeof(RBPF_Watchdog));

    if (config)
    {
        wd->config = *config;
    }
    else
    {
        rbpf_watchdog_default_config(&wd->config);
    }

    wd->ticks_since_last_reset = wd->config.min_ticks_between_resets;
    wd->last_reset_reason = RESET_NONE;
    wd->last_reset_severity = SEVERITY_NONE;
}

/*═══════════════════════════════════════════════════════════════════════════
 * HELPER FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

static int is_nan_f(float x) { return x != x; }
static int is_inf_f(float x) { return (x == INFINITY) || (x == -INFINITY); }

/*───────────────────────────────────────────────────────────────────────────
 * PCG32 RNG helpers (use RBPF's internal RNG for determinism)
 *───────────────────────────────────────────────────────────────────────────*/

static inline uint32_t wd_pcg32_next(uint64_t *state)
{
    uint64_t oldstate = *state;
    *state = oldstate * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static inline float wd_uniform(uint64_t *rng)
{
    return (float)(wd_pcg32_next(rng) >> 8) * (1.0f / 16777216.0f);
}

static inline float wd_gaussian(uint64_t *rng)
{
    /* Box-Muller transform */
    float u1 = wd_uniform(rng);
    float u2 = wd_uniform(rng);

    /* Avoid log(0) */
    if (u1 < 1e-10f)
        u1 = 1e-10f;

    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

/*───────────────────────────────────────────────────────────────────────────
 * Reset Storvik Sufficient Statistics to Priors
 *
 * CRITICAL: If parameters drifted to dangerous values (φ→1, σ→0),
 * resetting only state will cause immediate re-divergence.
 * This resets the learned parameters back to safe priors.
 *───────────────────────────────────────────────────────────────────────────*/

static void reset_storvik_params(RBPF_Extended *ext)
{
    if (!ext || !ext->storvik_initialized)
        return;

    ParamLearner *learner = &ext->storvik;

    /* Reset sufficient statistics to Priors for ALL particles/regimes */
    param_learn_broadcast_priors(learner);

    /*
     * Re-sync priors to particle arrays in RBPF core
     * so the next Predict step uses fresh values, not garbage.
     *
     * NOTE: This assumes RBPF_KSC has per-particle parameter arrays.
     * If your RBPF uses a different layout, adjust accordingly.
     */
    RBPF_KSC *rbpf = ext->rbpf;
    if (!rbpf)
        return;

    const int n = rbpf->n_particles;
    const int nr = rbpf->n_regimes;

    /* If RBPF has per-particle learned params, reset them too */
    /* This syncs the fresh priors to any cached param arrays */
    for (int r = 0; r < nr; r++)
    {
        RegimeParams params;
        param_learn_get_params(learner, -1, r, &params); /* Get prior */

        /* Update regime-level params if RBPF caches them */
        /* (Implementation depends on your RBPF_KSC layout) */
    }

    fprintf(stderr, "[WATCHDOG] Storvik parameters reset to priors.\n");
}

/**
 * Compute ESS from weights
 * ESS = 1 / Σ(w_i²) where w_i are normalized weights
 */
static float compute_ess(const float *weights, int n)
{
    if (!weights || n <= 0)
        return 0.0f;

    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++)
    {
        sum_sq += weights[i] * weights[i];
    }

    if (sum_sq < 1e-30f)
        return 0.0f;
    return 1.0f / sum_sq;
}

/**
 * Check for NaN/Inf in weight array
 */
static RBPF_ResetReason check_weights_valid(const float *weights, int n)
{
    if (!weights || n <= 0)
        return RESET_ZERO_WEIGHTS;

    float sum = 0.0f;
    int has_nan = 0;
    int has_inf = 0;

    for (int i = 0; i < n; i++)
    {
        if (is_nan_f(weights[i]))
            has_nan = 1;
        if (is_inf_f(weights[i]))
            has_inf = 1;
        sum += weights[i];
    }

    if (has_nan)
        return RESET_NAN_WEIGHT;
    if (has_inf)
        return RESET_INF_WEIGHT;
    if (sum < 1e-30f)
        return RESET_ZERO_WEIGHTS;

    return RESET_NONE;
}

/**
 * Check state estimates for validity
 */
static RBPF_ResetReason check_state_valid(float h_mean, float h_var, float max_abs)
{
    if (is_nan_f(h_mean) || is_nan_f(h_var))
        return RESET_STATE_NAN;
    if (is_inf_f(h_mean) || is_inf_f(h_var))
        return RESET_STATE_NAN;
    if (fabsf(h_mean) > max_abs)
        return RESET_STATE_EXPLOSION;

    return RESET_NONE;
}

/**
 * Check for regime deadlock
 *
 * Deadlock = We're in low regime (R0/R1) but seeing high-vol observations
 */
static int check_regime_deadlock(RBPF_Watchdog *wd, int current_regime,
                                 float obs, int n_regimes)
{
    if (!wd || wd->config.regime_deadlock_window <= 0)
        return 0;

    const int window = wd->config.regime_deadlock_window;

    /* Update circular buffers */
    wd->recent_returns[wd->recent_returns_idx % 32] = fabsf(obs);
    wd->recent_returns_idx++;

    wd->recent_regimes[wd->recent_regimes_idx % 32] = current_regime;
    wd->recent_regimes_idx++;

    /* Need enough history */
    if (wd->recent_returns_idx < window)
        return 0;

    /* Compute average absolute return over window */
    float avg_abs_ret = 0.0f;
    int start_idx = (wd->recent_returns_idx - window);
    for (int i = 0; i < window; i++)
    {
        avg_abs_ret += wd->recent_returns[(start_idx + i) % 32];
    }
    avg_abs_ret /= (float)window;

    /* Check if we've been in low regime the whole window */
    int all_low_regime = 1;
    start_idx = (wd->recent_regimes_idx - window);
    for (int i = 0; i < window; i++)
    {
        int r = wd->recent_regimes[(start_idx + i) % 32];
        if (r > 1)
        { /* Not in R0 or R1 */
            all_low_regime = 0;
            break;
        }
    }

    /* Deadlock: Low regime but high returns */
    if (all_low_regime && avg_abs_ret > wd->config.regime_deadlock_thresh)
    {
        return 1;
    }

    return 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN HEALTH CHECK
 *═══════════════════════════════════════════════════════════════════════════*/

RBPF_ResetReason rbpf_watchdog_check(RBPF_Extended *ext, RBPF_Watchdog *wd, float obs)
{
    if (!ext || !wd || !wd->config.enabled)
        return RESET_NONE;
    if (wd->circuit_breaker_tripped)
        return RESET_NONE; /* Already dead */

    wd->ticks_since_last_reset++;

    RBPF_KSC *rbpf = ext->rbpf;
    if (!rbpf)
        return RESET_NONE;

    const int n = rbpf->n_particles;
    const RBPF_WatchdogConfig *cfg = &wd->config;

    /*───────────────────────────────────────────────────────────────────────
     * CHECK 1: Weight validity (NaN, Inf, Zero)
     *─────────────────────────────────────────────────────────────────────*/
    RBPF_ResetReason weight_reason = check_weights_valid(rbpf->weight, n);
    if (weight_reason != RESET_NONE)
    {
        return weight_reason;
    }

    /*───────────────────────────────────────────────────────────────────────
     * CHECK 2: ESS collapse
     *─────────────────────────────────────────────────────────────────────*/
    float ess = compute_ess(rbpf->weight, n);
    float ess_ratio = ess / (float)n;

    if (ess_ratio < cfg->ess_critical_ratio)
    {
        return RESET_ESS_COLLAPSE;
    }

    if (ess_ratio < cfg->ess_warning_ratio)
    {
        wd->consecutive_ess_warnings++;
        if (wd->consecutive_ess_warnings % 10 == 1)
        { /* Rate limit logging */
            fprintf(stderr, "[WATCHDOG] ESS warning: %.1f%% (threshold: %.1f%%)\n",
                    ess_ratio * 100.0f, cfg->ess_warning_ratio * 100.0f);
        }
    }
    else
    {
        wd->consecutive_ess_warnings = 0;
    }

    /*───────────────────────────────────────────────────────────────────────
     * CHECK 3: Likelihood collapse
     *─────────────────────────────────────────────────────────────────────*/
    float log_lik = rbpf->output.marginal_likelihood;

    if (is_nan_f(log_lik))
    {
        return RESET_LIK_NAN;
    }

    if (log_lik < cfg->log_lik_floor)
    {
        return RESET_LIK_COLLAPSE;
    }

    if (log_lik < cfg->log_lik_warning)
    {
        wd->consecutive_lik_warnings++;
        if (wd->consecutive_lik_warnings % 10 == 1)
        {
            fprintf(stderr, "[WATCHDOG] Likelihood warning: %.2e (threshold: %.2e)\n",
                    log_lik, cfg->log_lik_warning);
        }
    }
    else
    {
        wd->consecutive_lik_warnings = 0;
    }

    /*───────────────────────────────────────────────────────────────────────
     * CHECK 4: State validity
     *─────────────────────────────────────────────────────────────────────*/
    float h_mean = rbpf->output.h_mean;
    float h_var = rbpf->output.h_var;

    RBPF_ResetReason state_reason = check_state_valid(h_mean, h_var, cfg->state_max_abs);
    if (state_reason != RESET_NONE)
    {
        return state_reason;
    }

    /*───────────────────────────────────────────────────────────────────────
     * CHECK 5: State divergence
     *
     * Compare estimated log-vol to observation-implied log-vol
     * y = log(r²) ≈ h + ε where ε ~ log(χ²(1))
     * Mean of log(χ²(1)) ≈ -1.27, so h_implied ≈ y + 1.27 ≈ log(r²)/2 roughly
     *─────────────────────────────────────────────────────────────────────*/
    if (fabsf(obs) > 1e-10f)
    {                                       /* Non-trivial observation */
        float y = logf(obs * obs + 1e-20f); /* log(r²) */
        float h_implied = y / 2.0f;         /* Rough approximation */

        float divergence = fabsf(h_mean - h_implied);
        float sigma_est = sqrtf(fmaxf(h_var, 0.01f));
        float divergence_sigma = divergence / sigma_est;

        if (divergence_sigma > cfg->state_divergence_sigma)
        {
            /* Track consecutive divergent observations in watchdog state */
            wd->consecutive_divergence_count++;

            if (wd->consecutive_divergence_count >= 3)
            { /* 3 consecutive */
                wd->consecutive_divergence_count = 0;
                return RESET_STATE_DIVERGENCE;
            }
        }
        else
        {
            /* Reset counter on good observation */
            wd->consecutive_divergence_count = 0;
        }
    }

    /*───────────────────────────────────────────────────────────────────────
     * CHECK 6: Regime deadlock
     *─────────────────────────────────────────────────────────────────────*/
    if (check_regime_deadlock(wd, rbpf->output.smoothed_regime, obs, rbpf->n_regimes))
    {
        return RESET_REGIME_DEADLOCK;
    }

    return RESET_NONE;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SEVERITY MAPPING
 *═══════════════════════════════════════════════════════════════════════════*/

RBPF_ResetSeverity rbpf_watchdog_get_severity(RBPF_ResetReason reason)
{
    switch (reason)
    {
    case RESET_NONE:
        return SEVERITY_NONE;

    /* Soft: Just need particle diversity */
    case RESET_ESS_COLLAPSE:
    case RESET_REGIME_DEADLOCK:
        return SEVERITY_SOFT;

    /* Medium: Reset state but keep parameters */
    case RESET_ZERO_WEIGHTS:
    case RESET_LIK_COLLAPSE:
    case RESET_STATE_DIVERGENCE:
    case RESET_MANUAL:
        return SEVERITY_MEDIUM;

    /* Hard: Full reset */
    case RESET_STATE_EXPLOSION:
    case RESET_LIK_NAN:
        return SEVERITY_HARD;

    /* Critical: Something very wrong */
    case RESET_NAN_WEIGHT:
    case RESET_INF_WEIGHT:
    case RESET_STATE_NAN:
        return SEVERITY_CRITICAL;

    default:
        return SEVERITY_MEDIUM;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * RESET EXECUTION
 *═══════════════════════════════════════════════════════════════════════════*/

int rbpf_watchdog_reset(RBPF_Extended *ext, RBPF_Watchdog *wd,
                        RBPF_ResetReason reason, RBPF_ResetEvent *event)
{
    if (!ext || !wd)
        return -1;
    if (reason == RESET_NONE)
        return 0;

    /*───────────────────────────────────────────────────────────────────────
     * Circuit breaker check
     *─────────────────────────────────────────────────────────────────────*/
    if (wd->circuit_breaker_tripped)
    {
        fprintf(stderr, "[WATCHDOG] CIRCUIT BREAKER TRIPPED - refusing reset\n");
        return -1;
    }

    /*───────────────────────────────────────────────────────────────────────
     * Cooldown check
     *─────────────────────────────────────────────────────────────────────*/
    if (wd->ticks_since_last_reset < wd->config.min_ticks_between_resets)
    {
        fprintf(stderr, "[WATCHDOG] Cooldown active (%d/%d ticks) - deferring reset\n",
                wd->ticks_since_last_reset, wd->config.min_ticks_between_resets);
        return 0; /* Not an error, just deferred */
    }

    RBPF_KSC *rbpf = ext->rbpf;
    RBPF_ResetSeverity severity = rbpf_watchdog_get_severity(reason);
    const RBPF_WatchdogConfig *cfg = &wd->config;

    /*───────────────────────────────────────────────────────────────────────
     * Capture state before reset (for logging/event)
     *─────────────────────────────────────────────────────────────────────*/
    float ess_before = compute_ess(rbpf->weight, rbpf->n_particles);
    float log_lik_before = rbpf->output.marginal_likelihood;
    float h_mean_before = rbpf->output.h_mean;
    float h_var_before = rbpf->output.h_var;
    int regime_before = rbpf->output.smoothed_regime;
    float hawkes_before = ext->hawkes.intensity;

    /*───────────────────────────────────────────────────────────────────────
     * Log the event
     *─────────────────────────────────────────────────────────────────────*/
    fprintf(stderr, "\n[WATCHDOG] ══════════════════════════════════════════\n");
    fprintf(stderr, "[WATCHDOG] RESET TRIGGERED: %s (severity: %s)\n",
            rbpf_watchdog_reason_str(reason), rbpf_watchdog_severity_str(severity));
    fprintf(stderr, "[WATCHDOG] Tick: %lu, Session resets: %d\n",
            (unsigned long)ext->tick_count, wd->session_resets + 1);
    fprintf(stderr, "[WATCHDOG] State before: h=%.3f±%.3f, regime=%d, ESS=%.1f\n",
            h_mean_before, sqrtf(h_var_before), regime_before, ess_before);
    fprintf(stderr, "[WATCHDOG] ══════════════════════════════════════════\n\n");

    /*───────────────────────────────────────────────────────────────────────
     * Populate event structure if provided
     *─────────────────────────────────────────────────────────────────────*/
    if (event)
    {
        event->tick = ext->tick_count;
        event->reason = reason;
        event->severity = severity;
        event->ess_before = ess_before;
        event->log_lik_before = log_lik_before;
        event->state_mean_before = h_mean_before;
        event->state_var_before = h_var_before;
        event->regime_before = regime_before;
        event->hawkes_intensity_before = hawkes_before;
    }

    /*───────────────────────────────────────────────────────────────────────
     * PERFORM RESET (severity-dependent)
     *─────────────────────────────────────────────────────────────────────*/
    const int n = rbpf->n_particles;

    switch (severity)
    {

    case SEVERITY_SOFT:
        /*───────────────────────────────────────────────────────────────────
         * SOFT: Re-diversify particles around current estimate
         * - Resample with replacement to fix weight degeneracy
         * - Add jitter to particle states
         *─────────────────────────────────────────────────────────────────*/
        {
            /* If weights are invalid, reset to uniform first */
            if (reason == RESET_NAN_WEIGHT || reason == RESET_ZERO_WEIGHTS)
            {
                float w_uniform = 1.0f / (float)n;
                for (int i = 0; i < n; i++)
                {
                    rbpf->weight[i] = w_uniform;
                }
            }

            /* Force resampling */
            rbpf_ksc_resample(rbpf);

            /* Add Gaussian jitter using internal RNG (deterministic) */
            float jitter_scale = 0.1f * sqrtf(fmaxf(h_var_before, 0.01f));

            /* Seed with tick + total_resets to ensure fresh entropy on repeated resets */
            uint64_t rng_state = ext->tick_count ^ (wd->total_resets * 0x9E3779B97F4A7C15ULL) ^ 0xDEADBEEF;

            for (int i = 0; i < n; i++)
            {
                float noise = wd_gaussian(&rng_state);
                rbpf->mu[i] += jitter_scale * noise;
            }

            /* Re-equalize weights */
            float w_uniform = 1.0f / (float)n;
            for (int i = 0; i < n; i++)
            {
                rbpf->weight[i] = w_uniform;
            }
        }
        break;

    case SEVERITY_MEDIUM:
        /*───────────────────────────────────────────────────────────────────
         * MEDIUM: Reset state distribution, keep parameters
         * - Re-initialize particles around best estimate
         * - Widen variance (uncertainty acknowledgment)
         * - Keep Storvik sufficient statistics
         *─────────────────────────────────────────────────────────────────*/
        {
            /* Use best available estimate as center */
            float h_center = h_mean_before;
            if (is_nan_f(h_center) || is_inf_f(h_center))
            {
                h_center = -4.0f; /* Fallback to ~2% vol */
            }

            /* Widened variance (with cap to prevent explosion) */
            float h_std = cfg->reset_variance_scale * sqrtf(fmaxf(h_var_before, 0.1f));
            if (is_nan_f(h_std) || h_std < 0.1f)
            {
                h_std = 0.5f; /* Fallback */
            }
            if (h_std > 2.0f)
            {
                h_std = 2.0f; /* Cap explosion */
            }

            /* Seed with tick + total_resets for fresh entropy on repeated resets */
            uint64_t rng_state = ext->tick_count ^ (wd->total_resets * 0x9E3779B97F4A7C15ULL) ^ 0xCAFEBABE;

            /* Re-initialize particles */
            for (int i = 0; i < n; i++)
            {
                float noise = wd_gaussian(&rng_state);

                rbpf->mu[i] = h_center + h_std * noise;
                rbpf->var[i] = h_std * h_std;
                rbpf->weight[i] = 1.0f / (float)n;

                /* Regime diversity (don't lock all particles in one regime) */
                rbpf->regime[i] = i % rbpf->n_regimes;
            }
        }
        break;

    case SEVERITY_HARD:
    case SEVERITY_CRITICAL:
        /*───────────────────────────────────────────────────────────────────
         * HARD/CRITICAL: Full reset
         * - Re-initialize everything
         * - Optionally reset learned parameters (Storvik)
         *
         * CRITICAL FIX: If params drifted to dangerous values (φ→1, σ→0),
         * resetting only state causes immediate re-divergence ("reset storm").
         * Must also reset Storvik sufficient statistics.
         *─────────────────────────────────────────────────────────────────*/
        {
            /* Full state reset */
            float h_init = -4.0f; /* ~2% vol */
            float P_init = 1.0f;  /* Wide uncertainty */

            /* Seed with tick + total_resets for fresh entropy on repeated resets */
            uint64_t rng_state = ext->tick_count ^ (wd->total_resets * 0x9E3779B97F4A7C15ULL) ^ 0xDEADC0DE;

            for (int i = 0; i < n; i++)
            {
                float noise = wd_gaussian(&rng_state);

                rbpf->mu[i] = h_init + sqrtf(P_init) * noise;
                rbpf->var[i] = P_init;
                rbpf->weight[i] = 1.0f / (float)n;
                rbpf->regime[i] = i % rbpf->n_regimes;
            }

            /* Reset Storvik parameters if configured */
            if (!cfg->preserve_learned_params)
            {
                reset_storvik_params(ext);
            }
        }
        break;

    default:
        break;
    }

    /*───────────────────────────────────────────────────────────────────────
     * Clear Hawkes intensity if configured
     *─────────────────────────────────────────────────────────────────────*/
    if (cfg->clear_hawkes_on_reset && ext->hawkes.enabled)
    {
        ext->hawkes.intensity = ext->hawkes.mu;
        ext->hawkes.intensity_prev = ext->hawkes.mu;
        ext->hawkes.lut_dirty = 0;

        /* Restore base transitions */
        if (ext->hawkes.lut_dirty)
        {
            rbpf_ksc_build_transition_lut(rbpf, ext->base_trans_matrix);
        }
    }

    /*───────────────────────────────────────────────────────────────────────
     * Update watchdog state
     *─────────────────────────────────────────────────────────────────────*/
    wd->ticks_since_last_reset = 0;
    wd->total_resets++;
    wd->session_resets++;
    wd->last_reset_reason = reason;
    wd->last_reset_severity = severity;
    wd->last_reset_tick = ext->tick_count;
    wd->last_reset_ess = ess_before;
    wd->last_reset_log_lik = log_lik_before;
    wd->last_reset_state_mean = h_mean_before;

    /* Clear rolling buffers */
    wd->recent_returns_idx = 0;
    wd->recent_regimes_idx = 0;
    wd->consecutive_ess_warnings = 0;
    wd->consecutive_lik_warnings = 0;
    wd->consecutive_divergence_count = 0;

    /*───────────────────────────────────────────────────────────────────────
     * Check circuit breaker
     *─────────────────────────────────────────────────────────────────────*/
    if (cfg->max_resets_per_session > 0 &&
        wd->session_resets >= cfg->max_resets_per_session)
    {
        wd->circuit_breaker_tripped = 1;
        fprintf(stderr, "\n[WATCHDOG] *** CIRCUIT BREAKER TRIPPED ***\n");
        fprintf(stderr, "[WATCHDOG] Max resets (%d) exceeded. Filter halted.\n",
                cfg->max_resets_per_session);
        fprintf(stderr, "[WATCHDOG] Manual intervention required.\n\n");
    }

    /*───────────────────────────────────────────────────────────────────────
     * Invoke callback if registered
     *─────────────────────────────────────────────────────────────────────*/
    if (wd->on_reset_callback)
    {
        wd->on_reset_callback(reason, severity, wd->callback_user_data);
    }

    return 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * CONVENIENCE FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

int rbpf_watchdog_manual_reset(RBPF_Extended *ext, RBPF_Watchdog *wd)
{
    return rbpf_watchdog_reset(ext, wd, RESET_MANUAL, NULL);
}

void rbpf_watchdog_set_callback(RBPF_Watchdog *wd,
                                void (*callback)(RBPF_ResetReason, RBPF_ResetSeverity, void *),
                                void *user_data)
{
    if (!wd)
        return;
    wd->on_reset_callback = callback;
    wd->callback_user_data = user_data;
}

void rbpf_watchdog_new_session(RBPF_Watchdog *wd)
{
    if (!wd)
        return;
    wd->session_resets = 0;
    wd->circuit_breaker_tripped = 0;
    wd->ticks_since_last_reset = wd->config.min_ticks_between_resets;
    wd->recent_returns_idx = 0;
    wd->recent_regimes_idx = 0;
    wd->consecutive_ess_warnings = 0;
    wd->consecutive_lik_warnings = 0;
    wd->consecutive_divergence_count = 0;
}

int rbpf_watchdog_is_circuit_breaker_tripped(const RBPF_Watchdog *wd)
{
    return wd ? wd->circuit_breaker_tripped : 0;
}

const char *rbpf_watchdog_reason_str(RBPF_ResetReason reason)
{
    if (reason < 0 || reason >= RESET_REASON_COUNT)
        return "UNKNOWN";
    return REASON_STRINGS[reason];
}

const char *rbpf_watchdog_severity_str(RBPF_ResetSeverity severity)
{
    if (severity < 0 || severity > SEVERITY_CRITICAL)
        return "UNKNOWN";
    return SEVERITY_STRINGS[severity];
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_watchdog_print_status(const RBPF_Watchdog *wd)
{
    if (!wd)
        return;

    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║              RBPF WATCHDOG STATUS                        ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║ Enabled:              %s                                 ║\n",
           wd->config.enabled ? "YES" : "NO ");
    printf("║ Circuit Breaker:      %s                                 ║\n",
           wd->circuit_breaker_tripped ? "TRIPPED!" : "OK      ");
    printf("║ Total Resets:         %-6d                             ║\n",
           wd->total_resets);
    printf("║ Session Resets:       %-6d (max: %d)                    ║\n",
           wd->session_resets, wd->config.max_resets_per_session);
    printf("║ Ticks Since Reset:    %-6d                             ║\n",
           wd->ticks_since_last_reset);
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║ Last Reset Reason:    %-20s             ║\n",
           rbpf_watchdog_reason_str(wd->last_reset_reason));
    printf("║ Last Reset Severity:  %-10s                         ║\n",
           rbpf_watchdog_severity_str(wd->last_reset_severity));
    printf("║ Last Reset Tick:      %-10lu                         ║\n",
           (unsigned long)wd->last_reset_tick);
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║ ESS Critical:         %.1f%%                              ║\n",
           wd->config.ess_critical_ratio * 100.0f);
    printf("║ Log-Lik Floor:        %.2e                           ║\n",
           wd->config.log_lik_floor);
    printf("║ State Max |h|:        %.1f                               ║\n",
           wd->config.state_max_abs);
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
}

void rbpf_watchdog_print_event(const RBPF_ResetEvent *event)
{
    if (!event)
        return;

    printf("\n┌──────────────────────────────────────────────────────────┐\n");
    printf("│              RESET EVENT DETAILS                         │\n");
    printf("├──────────────────────────────────────────────────────────┤\n");
    printf("│ Tick:       %-10lu                                   │\n",
           (unsigned long)event->tick);
    printf("│ Reason:     %-20s                       │\n",
           rbpf_watchdog_reason_str(event->reason));
    printf("│ Severity:   %-10s                                   │\n",
           rbpf_watchdog_severity_str(event->severity));
    printf("├──────────────────────────────────────────────────────────┤\n");
    printf("│ STATE BEFORE RESET:                                      │\n");
    printf("│   ESS:        %.1f                                       │\n",
           event->ess_before);
    printf("│   Log-Lik:    %.4e                                   │\n",
           event->log_lik_before);
    printf("│   h_mean:     %.4f                                       │\n",
           event->state_mean_before);
    printf("│   h_var:      %.4f                                       │\n",
           event->state_var_before);
    printf("│   Regime:     %d                                          │\n",
           event->regime_before);
    printf("│   Hawkes λ:   %.4f                                       │\n",
           event->hawkes_intensity_before);
    printf("└──────────────────────────────────────────────────────────┘\n\n");
}