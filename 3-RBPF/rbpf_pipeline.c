/*=============================================================================
 * RBPF Pipeline: Unified Change Detection + Volatility Tracking
 *
 * Replaces: BOCPD + PF stack
 *
 * Single filter provides:
 *   - Volatility estimation (for Kelly sizing)
 *   - Change detection (for risk management)
 *   - Regime identification (for parameter selection)
 *   - Uncertainty quantification (for position scaling)
 *
 * Usage:
 *   RBPF_Pipeline *pipe = rbpf_pipeline_create(config);
 *
 *   for each tick:
 *       rbpf_pipeline_step(pipe, ssa_cleaned_return, &signal);
 *       kelly_fraction = kelly_compute(signal.vol_forecast, ...) * signal.position_scale;
 *
 * Author: RBPF-KSC Project
 * License: MIT
 *===========================================================================*/

#include "rbpf_ksc.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*─────────────────────────────────────────────────────────────────────────────
 * PIPELINE STATE (concrete definition of opaque handle)
 *───────────────────────────────────────────────────────────────────────────*/

struct RBPF_Pipeline
{
    RBPF_KSC *rbpf;             /* Underlying filter */
    RBPF_PipelineConfig config; /* Configuration snapshot */

    /* State */
    int tick_count;
    int initialized;

    /* Change detection state */
    rbpf_real_t baseline_vol; /* Reference volatility */
    int last_regime;          /* For regime change detection */
    int ticks_in_regime;      /* Stability counter */

    /* Confirmation window counters (NEW - reduces false positives) */
    int consecutive_surprise;  /* Consecutive ticks above surprise threshold */
    int consecutive_vol_spike; /* Consecutive ticks above vol_ratio threshold */
};

/*─────────────────────────────────────────────────────────────────────────────
 * DEFAULT CONFIGURATION
 *───────────────────────────────────────────────────────────────────────────*/

RBPF_PipelineConfig rbpf_pipeline_default_config(void)
{
    RBPF_PipelineConfig cfg = {0};

    /* Filter size */
    cfg.n_particles = 200;
    cfg.n_regimes = 4;

    /* Regime parameters (typical equity intraday) - STABLE VALUES */
    cfg.theta[0] = RBPF_REAL(0.05);
    cfg.mu_vol[0] = RBPF_REAL(-4.6);
    cfg.sigma_vol[0] = RBPF_REAL(0.05);
    cfg.theta[1] = RBPF_REAL(0.08);
    cfg.mu_vol[1] = RBPF_REAL(-3.5);
    cfg.sigma_vol[1] = RBPF_REAL(0.10);
    cfg.theta[2] = RBPF_REAL(0.12);
    cfg.mu_vol[2] = RBPF_REAL(-2.5);
    cfg.sigma_vol[2] = RBPF_REAL(0.20);
    cfg.theta[3] = RBPF_REAL(0.15);
    cfg.mu_vol[3] = RBPF_REAL(-1.6); /* STABLE: -1.6 (~20% vol) */
    cfg.sigma_vol[3] = RBPF_REAL(0.30);

    /* Transition matrix (moderately sticky) */
    rbpf_real_t trans[16] = {
        RBPF_REAL(0.92), RBPF_REAL(0.05), RBPF_REAL(0.02), RBPF_REAL(0.01),
        RBPF_REAL(0.05), RBPF_REAL(0.88), RBPF_REAL(0.05), RBPF_REAL(0.02),
        RBPF_REAL(0.02), RBPF_REAL(0.05), RBPF_REAL(0.88), RBPF_REAL(0.05),
        RBPF_REAL(0.01), RBPF_REAL(0.02), RBPF_REAL(0.05), RBPF_REAL(0.92)};
    memcpy(cfg.transition, trans, sizeof(trans));

    /* Change detection - balanced for moderate events + low FP */
    cfg.surprise_minor = RBPF_REAL(3.5);
    cfg.surprise_major = RBPF_REAL(5.5);
    cfg.vol_ratio_minor = RBPF_REAL(1.75);
    cfg.vol_ratio_major = RBPF_REAL(2.25);

    /* NEW: Confirmation window (reduces false positives)
     * Require N consecutive ticks above threshold before triggering.
     * Single spike = noise, sustained spike = real regime change.
     *
     * EXCEPTION: Extreme events (>8σ) bypass confirmation entirely.
     */
    cfg.confirm_minor = 3;      /* 3 consecutive ticks for minor alert */
    cfg.confirm_major = 2;      /* 2 consecutive ticks for major alert */
    cfg.surprise_extreme = 8.0; /* Extreme threshold - bypasses confirmation */

    /* Position scaling */
    cfg.scale_on_minor = RBPF_REAL(0.5);
    cfg.scale_on_major = RBPF_REAL(0.25);
    cfg.scale_low_confidence = RBPF_REAL(0.7);
    cfg.confidence_threshold = RBPF_REAL(0.6);

    /* Smoothing */
    cfg.smooth_lag = 5;
    cfg.regime_hold_ticks = 5;

    /* Learning off by default */
    cfg.enable_learning = 0;
    cfg.learning_shrinkage = RBPF_REAL(0.95);
    cfg.learning_warmup = 100;

    return cfg;
}

/*─────────────────────────────────────────────────────────────────────────────
 * CREATE / DESTROY
 *───────────────────────────────────────────────────────────────────────────*/

RBPF_Pipeline *rbpf_pipeline_create(const RBPF_PipelineConfig *config)
{
    RBPF_Pipeline *pipe = (RBPF_Pipeline *)calloc(1, sizeof(RBPF_Pipeline));
    if (!pipe)
        return NULL;

    /* Use provided config or defaults */
    if (config)
    {
        pipe->config = *config;
    }
    else
    {
        pipe->config = rbpf_pipeline_default_config();
    }

    const RBPF_PipelineConfig *cfg = &pipe->config;

    /* Create underlying RBPF */
    pipe->rbpf = rbpf_ksc_create(cfg->n_particles, cfg->n_regimes);
    if (!pipe->rbpf)
    {
        free(pipe);
        return NULL;
    }

    /* Configure regime parameters */
    for (int r = 0; r < cfg->n_regimes; r++)
    {
        rbpf_ksc_set_regime_params(pipe->rbpf, r,
                                   cfg->theta[r],
                                   cfg->mu_vol[r],
                                   cfg->sigma_vol[r]);
    }

    /* Build transition LUT */
    rbpf_ksc_build_transition_lut(pipe->rbpf, cfg->transition);

    /* Regularization (sensible defaults) */
    rbpf_ksc_set_regularization(pipe->rbpf, RBPF_REAL(0.02), RBPF_REAL(0.001));
    rbpf_ksc_set_regime_diversity(pipe->rbpf, cfg->n_particles / 25, RBPF_REAL(0.01));

    /* Smoothing */
    rbpf_ksc_set_fixed_lag_smoothing(pipe->rbpf, cfg->smooth_lag);
    rbpf_ksc_set_regime_smoothing(pipe->rbpf, cfg->regime_hold_ticks, RBPF_REAL(0.7));

    /* Liu-West learning */
    if (cfg->enable_learning)
    {
        rbpf_ksc_enable_liu_west(pipe->rbpf, cfg->learning_shrinkage, cfg->learning_warmup);
    }

    pipe->initialized = 0;
    pipe->tick_count = 0;

    /* Initialize confirmation counters */
    pipe->consecutive_surprise = 0;
    pipe->consecutive_vol_spike = 0;

    return pipe;
}

void rbpf_pipeline_destroy(RBPF_Pipeline *pipe)
{
    if (!pipe)
        return;
    if (pipe->rbpf)
        rbpf_ksc_destroy(pipe->rbpf);
    free(pipe);
}

/*─────────────────────────────────────────────────────────────────────────────
 * INITIALIZATION
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_pipeline_init(RBPF_Pipeline *pipe, rbpf_real_t initial_vol)
{
    rbpf_real_t log_vol = rbpf_log(initial_vol);
    rbpf_ksc_init(pipe->rbpf, log_vol, RBPF_REAL(0.1));
    rbpf_ksc_warmup(pipe->rbpf);

    pipe->baseline_vol = initial_vol;
    pipe->last_regime = 0;
    pipe->ticks_in_regime = 0;
    pipe->tick_count = 0;
    pipe->initialized = 1;

    /* Reset confirmation counters */
    pipe->consecutive_surprise = 0;
    pipe->consecutive_vol_spike = 0;
}

/*─────────────────────────────────────────────────────────────────────────────
 * MAIN STEP: Process one SSA-cleaned return
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_pipeline_step(RBPF_Pipeline *pipe, rbpf_real_t ssa_return, RBPF_Signal *sig)
{
    const RBPF_PipelineConfig *cfg = &pipe->config;

    /* Initialize on first call if needed */
    if (!pipe->initialized)
    {
        rbpf_pipeline_init(pipe, RBPF_REAL(0.01));
    }

    /* Run RBPF */
    RBPF_KSC_Output out;
    rbpf_ksc_step(pipe->rbpf, ssa_return, &out);
    pipe->tick_count++;

    /*========================================================================
     * VOLATILITY OUTPUT
     *======================================================================*/
    sig->vol_forecast = out.vol_mean;
    sig->log_vol = out.log_vol_mean;

    /* Uncertainty: use smooth estimates if available */
    if (out.smooth_valid)
    {
        sig->vol_uncertainty = rbpf_sqrt(out.log_vol_var_smooth);
    }
    else
    {
        sig->vol_uncertainty = rbpf_sqrt(out.log_vol_var);
    }

    /*========================================================================
     * REGIME OUTPUT
     *======================================================================*/
    sig->regime = out.smoothed_regime;
    sig->regime_confidence = out.regime_confidence;
    for (int r = 0; r < cfg->n_regimes; r++)
    {
        sig->regime_probs[r] = out.regime_probs_smooth[r];
    }

    /* Track regime stability */
    if (out.smoothed_regime == pipe->last_regime)
    {
        pipe->ticks_in_regime++;
    }
    else
    {
        pipe->last_regime = out.smoothed_regime;
        pipe->ticks_in_regime = 1;
    }

    /*========================================================================
     * CHANGE DETECTION (with confirmation window)
     *
     * NEW: Require N consecutive ticks above threshold to trigger.
     * This eliminates single-tick false positives from noise.
     *======================================================================*/
    sig->surprise = out.surprise;
    sig->vol_ratio = out.vol_ratio;
    sig->regime_entropy = out.regime_entropy;
    sig->ess = out.ess;
    sig->tick = pipe->tick_count;

    /* Track raw signal state (before confirmation) */
    int raw_surprise_major = (out.surprise >= cfg->surprise_major);
    int raw_surprise_minor = (out.surprise >= cfg->surprise_minor);
    int raw_surprise_extreme = (out.surprise >= cfg->surprise_extreme); /* NEW: extreme bypass */
    int raw_vol_major = (out.vol_ratio >= cfg->vol_ratio_major);
    int raw_vol_minor = (out.vol_ratio >= cfg->vol_ratio_minor);

    /* Update consecutive counters */
    if (raw_surprise_major || raw_surprise_minor)
    {
        pipe->consecutive_surprise++;
    }
    else
    {
        pipe->consecutive_surprise = 0;
    }

    if (raw_vol_major || raw_vol_minor)
    {
        pipe->consecutive_vol_spike++;
    }
    else
    {
        pipe->consecutive_vol_spike = 0;
    }

    /* Apply confirmation window - only trigger after N consecutive ticks
     * EXCEPTION: Extreme events (>8σ) bypass confirmation for safety */
    int confirmed_surprise_major = raw_surprise_extreme || /* Extreme bypasses */
                                   (raw_surprise_major &&
                                    (pipe->consecutive_surprise >= cfg->confirm_major));
    int confirmed_surprise_minor = raw_surprise_minor &&
                                   (pipe->consecutive_surprise >= cfg->confirm_minor);
    int confirmed_vol_major = raw_vol_major &&
                              (pipe->consecutive_vol_spike >= cfg->confirm_major);
    int confirmed_vol_minor = raw_vol_minor &&
                              (pipe->consecutive_vol_spike >= cfg->confirm_minor);

    /* Classify change (using CONFIRMED signals) */
    int vol_spike = 0;
    int regime_shift = 0;

    if (confirmed_surprise_major || confirmed_vol_major)
    {
        sig->change_detected = 2; /* Major - confirmed */
        vol_spike = confirmed_vol_major;
    }
    else if (confirmed_surprise_minor || confirmed_vol_minor)
    {
        sig->change_detected = 1; /* Minor - confirmed */
        vol_spike = confirmed_vol_minor;
    }
    else
    {
        sig->change_detected = 0;
    }

    /* Regime shift detection */
    if (out.regime_changed || (pipe->ticks_in_regime == 1 && pipe->tick_count > 1))
    {
        regime_shift = 1;
        if (sig->change_detected < 1)
            sig->change_detected = 1;
    }

    /* Change type */
    if (vol_spike && regime_shift)
    {
        sig->change_type = 3; /* Both */
    }
    else if (regime_shift)
    {
        sig->change_type = 2; /* Regime shift */
    }
    else if (vol_spike)
    {
        sig->change_type = 1; /* Vol spike */
    }
    else
    {
        sig->change_type = 0; /* None */
    }

    /*========================================================================
     * POSITION SCALING (Risk Management)
     *
     * This is the key output for Kelly integration.
     * Scale down positions when:
     *   - Major change detected (high surprise or vol spike)
     *   - Regime uncertain (low confidence)
     *   - Recent regime shift (need confirmation)
     *======================================================================*/

    rbpf_real_t scale = RBPF_REAL(1.0);
    int action = 0;

    /* Major change: aggressive reduction */
    if (sig->change_detected == 2)
    {
        scale = cfg->scale_on_major;
        action = 2; /* Exit/hedge */
    }
    /* Minor change: moderate reduction */
    else if (sig->change_detected == 1)
    {
        scale = cfg->scale_on_minor;
        action = 1; /* Reduce */
    }

    /* Low confidence: additional reduction */
    if (sig->regime_confidence < cfg->confidence_threshold)
    {
        scale *= cfg->scale_low_confidence;
        if (action < 1)
            action = 1;
    }

    /* Recent regime shift: wait for confirmation */
    if (pipe->ticks_in_regime < cfg->regime_hold_ticks)
    {
        scale *= RBPF_REAL(0.8); /* 20% reduction during transition */
    }

    /* High regime entropy: uncertain state */
    if (sig->regime_entropy > RBPF_REAL(1.2))
    {
        scale *= RBPF_REAL(0.9);
    }

    /* Floor at 10% */
    if (scale < RBPF_REAL(0.1))
        scale = RBPF_REAL(0.1);

    sig->position_scale = scale;
    sig->action = action;
}

/*─────────────────────────────────────────────────────────────────────────────
 * CONVENIENCE: Step with lookahead (APF mode)
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_pipeline_step_apf(RBPF_Pipeline *pipe,
                            rbpf_real_t ssa_return_current,
                            rbpf_real_t ssa_return_next,
                            RBPF_Signal *sig)
{
    const RBPF_PipelineConfig *cfg = &pipe->config;

    if (!pipe->initialized)
    {
        rbpf_pipeline_init(pipe, RBPF_REAL(0.01));
    }

    /* Use adaptive APF */
    RBPF_KSC_Output out;
    rbpf_ksc_step_adaptive(pipe->rbpf, ssa_return_current, ssa_return_next, &out);
    pipe->tick_count++;

    /* Rest of processing identical to standard step */
    sig->vol_forecast = out.vol_mean;
    sig->log_vol = out.log_vol_mean;
    sig->vol_uncertainty = out.smooth_valid ? rbpf_sqrt(out.log_vol_var_smooth) : rbpf_sqrt(out.log_vol_var);

    sig->regime = out.smoothed_regime;
    sig->regime_confidence = out.regime_confidence;
    for (int r = 0; r < cfg->n_regimes; r++)
    {
        sig->regime_probs[r] = out.regime_probs_smooth[r];
    }

    if (out.smoothed_regime == pipe->last_regime)
    {
        pipe->ticks_in_regime++;
    }
    else
    {
        pipe->last_regime = out.smoothed_regime;
        pipe->ticks_in_regime = 1;
    }

    sig->surprise = out.surprise;
    sig->vol_ratio = out.vol_ratio;
    sig->regime_entropy = out.regime_entropy;
    sig->ess = out.ess;
    sig->tick = pipe->tick_count;

    /* Change detection with confirmation (same logic as standard step) */
    int raw_surprise_major = (out.surprise >= cfg->surprise_major);
    int raw_surprise_minor = (out.surprise >= cfg->surprise_minor);
    int raw_surprise_extreme = (out.surprise >= cfg->surprise_extreme); /* Extreme bypass */
    int raw_vol_major = (out.vol_ratio >= cfg->vol_ratio_major);
    int raw_vol_minor = (out.vol_ratio >= cfg->vol_ratio_minor);

    if (raw_surprise_major || raw_surprise_minor)
    {
        pipe->consecutive_surprise++;
    }
    else
    {
        pipe->consecutive_surprise = 0;
    }

    if (raw_vol_major || raw_vol_minor)
    {
        pipe->consecutive_vol_spike++;
    }
    else
    {
        pipe->consecutive_vol_spike = 0;
    }

    /* Extreme events bypass confirmation for safety */
    int confirmed_surprise_major = raw_surprise_extreme ||
                                   (raw_surprise_major &&
                                    (pipe->consecutive_surprise >= cfg->confirm_major));
    int confirmed_surprise_minor = raw_surprise_minor &&
                                   (pipe->consecutive_surprise >= cfg->confirm_minor);
    int confirmed_vol_major = raw_vol_major &&
                              (pipe->consecutive_vol_spike >= cfg->confirm_major);
    int confirmed_vol_minor = raw_vol_minor &&
                              (pipe->consecutive_vol_spike >= cfg->confirm_minor);

    int vol_spike = 0;
    int regime_shift = 0;

    if (confirmed_surprise_major || confirmed_vol_major)
    {
        sig->change_detected = 2;
        vol_spike = confirmed_vol_major;
    }
    else if (confirmed_surprise_minor || confirmed_vol_minor)
    {
        sig->change_detected = 1;
        vol_spike = confirmed_vol_minor;
    }
    else
    {
        sig->change_detected = 0;
    }

    if (out.regime_changed || (pipe->ticks_in_regime == 1 && pipe->tick_count > 1))
    {
        regime_shift = 1;
        if (sig->change_detected < 1)
            sig->change_detected = 1;
    }

    if (vol_spike && regime_shift)
    {
        sig->change_type = 3;
    }
    else if (regime_shift)
    {
        sig->change_type = 2;
    }
    else if (vol_spike)
    {
        sig->change_type = 1;
    }
    else
    {
        sig->change_type = 0;
    }

    /* Position scaling */
    rbpf_real_t scale = RBPF_REAL(1.0);
    int action = 0;

    if (sig->change_detected == 2)
    {
        scale = cfg->scale_on_major;
        action = 2;
    }
    else if (sig->change_detected == 1)
    {
        scale = cfg->scale_on_minor;
        action = 1;
    }

    if (sig->regime_confidence < cfg->confidence_threshold)
    {
        scale *= cfg->scale_low_confidence;
        if (action < 1)
            action = 1;
    }

    if (pipe->ticks_in_regime < cfg->regime_hold_ticks)
    {
        scale *= RBPF_REAL(0.8);
    }

    if (sig->regime_entropy > RBPF_REAL(1.2))
    {
        scale *= RBPF_REAL(0.9);
    }

    if (scale < RBPF_REAL(0.1))
        scale = RBPF_REAL(0.1);

    sig->position_scale = scale;
    sig->action = action;
}

/*─────────────────────────────────────────────────────────────────────────────
 * ACCESSORS
 *───────────────────────────────────────────────────────────────────────────*/

int rbpf_pipeline_get_tick(const RBPF_Pipeline *pipe)
{
    return pipe->tick_count;
}

int rbpf_pipeline_get_regime(const RBPF_Pipeline *pipe)
{
    return pipe->last_regime;
}

rbpf_real_t rbpf_pipeline_get_baseline_vol(const RBPF_Pipeline *pipe)
{
    return pipe->baseline_vol;
}

void rbpf_pipeline_set_baseline_vol(RBPF_Pipeline *pipe, rbpf_real_t vol)
{
    pipe->baseline_vol = vol;
}

/* Get learned parameters (if Liu-West enabled) */
void rbpf_pipeline_get_learned_params(const RBPF_Pipeline *pipe, int regime,
                                      rbpf_real_t *mu_vol, rbpf_real_t *sigma_vol)
{
    rbpf_ksc_get_learned_params(pipe->rbpf, regime, mu_vol, sigma_vol);
}

/*─────────────────────────────────────────────────────────────────────────────
 * CONFIGURATION UPDATE (Runtime tuning)
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_pipeline_set_thresholds(RBPF_Pipeline *pipe,
                                  rbpf_real_t surprise_minor, rbpf_real_t surprise_major,
                                  rbpf_real_t vol_ratio_minor, rbpf_real_t vol_ratio_major)
{
    pipe->config.surprise_minor = surprise_minor;
    pipe->config.surprise_major = surprise_major;
    pipe->config.vol_ratio_minor = vol_ratio_minor;
    pipe->config.vol_ratio_major = vol_ratio_major;
}

void rbpf_pipeline_set_scaling(RBPF_Pipeline *pipe,
                               rbpf_real_t scale_minor, rbpf_real_t scale_major,
                               rbpf_real_t scale_low_conf, rbpf_real_t conf_threshold)
{
    pipe->config.scale_on_minor = scale_minor;
    pipe->config.scale_on_major = scale_major;
    pipe->config.scale_low_confidence = scale_low_conf;
    pipe->config.confidence_threshold = conf_threshold;
}

/*─────────────────────────────────────────────────────────────────────────────
 * DEBUG
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_pipeline_print_config(const RBPF_Pipeline *pipe)
{
    const RBPF_PipelineConfig *cfg = &pipe->config;

    printf("RBPF Pipeline Configuration:\n");
    printf("  Particles: %d\n", cfg->n_particles);
    printf("  Regimes:   %d\n", cfg->n_regimes);
    printf("\n  Regime Parameters:\n");
    printf("  %-8s %8s %8s %8s\n", "Regime", "theta", "mu_vol", "sigma_vol");
    for (int r = 0; r < cfg->n_regimes; r++)
    {
        printf("  %-8d %8.4f %8.4f %8.4f\n",
               r, cfg->theta[r], cfg->mu_vol[r], cfg->sigma_vol[r]);
    }
    printf("\n  Change Detection Thresholds:\n");
    printf("    Surprise: minor=%.2f, major=%.2f\n",
           cfg->surprise_minor, cfg->surprise_major);
    printf("    Vol ratio: minor=%.2f, major=%.2f\n",
           cfg->vol_ratio_minor, cfg->vol_ratio_major);
    printf("\n  Confirmation Window:\n");
    printf("    Minor: %d consecutive ticks\n", cfg->confirm_minor);
    printf("    Major: %d consecutive ticks\n", cfg->confirm_major);
    printf("\n  Position Scaling:\n");
    printf("    On minor change: %.2f\n", cfg->scale_on_minor);
    printf("    On major change: %.2f\n", cfg->scale_on_major);
    printf("    Low confidence:  %.2f (threshold: %.2f)\n",
           cfg->scale_low_confidence, cfg->confidence_threshold);
    printf("\n  Smoothing:\n");
    printf("    Lag: %d ticks\n", cfg->smooth_lag);
    printf("    Regime hold: %d ticks\n", cfg->regime_hold_ticks);
    printf("\n  Learning: %s\n", cfg->enable_learning ? "enabled" : "disabled");
}

void rbpf_pipeline_print_signal(const RBPF_Signal *sig)
{
    const char *change_str[] = {"none", "minor", "MAJOR"};
    const char *type_str[] = {"none", "vol_spike", "regime_shift", "both"};
    const char *action_str[] = {"normal", "reduce", "EXIT"};

    printf("Signal[t=%d]: vol=%.4f regime=%d conf=%.2f | "
           "change=%s type=%s | scale=%.2f action=%s\n",
           sig->tick, sig->vol_forecast, sig->regime, sig->regime_confidence,
           change_str[sig->change_detected], type_str[sig->change_type],
           sig->position_scale, action_str[sig->action]);
}