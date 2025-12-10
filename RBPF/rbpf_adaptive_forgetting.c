/**
 * @file rbpf_adaptive_forgetting.c
 * @brief Adaptive forgetting based on predictive surprise
 *
 * Implements West & Harrison's Bayesian Intervention Analysis:
 *   - Monitor: Track one-step-ahead forecast error (predictive surprise)
 *   - Intervene: When error spikes, reduce λ to accelerate adaptation
 *
 * The key insight: Low likelihood means the current parameter set θ is
 * unlikely to have generated this observation. We should "forget" old
 * sufficient statistics faster to adapt to the new regime.
 *
 * Reference: West & Harrison (1997) "Bayesian Forecasting and Dynamic Models"
 */

#include "rbpf_ksc_param_integration.h"
#include "rbpf_param_learn.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/*============================================================================
 * CONSTANTS
 *============================================================================*/

/* Default regime λ values (GARCH-inspired persistence) */
static const rbpf_real_t DEFAULT_LAMBDA_REGIME[RBPF_MAX_REGIMES] = {
    RBPF_REAL(0.999), /* R0 (Calm): Very long memory, N_eff ≈ 1000 */
    RBPF_REAL(0.998), /* R1 (Normal): Long memory, N_eff ≈ 500 */
    RBPF_REAL(0.995), /* R2 (Elevated): Medium memory, N_eff ≈ 200 */
    RBPF_REAL(0.990), /* R3 (Crisis): Shorter memory, N_eff ≈ 100 */
    RBPF_REAL(0.985), /* R4+: Even shorter for extreme regimes */
    RBPF_REAL(0.985),
    RBPF_REAL(0.985),
    RBPF_REAL(0.985)};

/* Sigmoid defaults */
#define DEFAULT_SIGMOID_CENTER RBPF_REAL(2.0)    /* 2σ for half-response */
#define DEFAULT_SIGMOID_STEEPNESS RBPF_REAL(2.0) /* Moderate sharpness */
#define DEFAULT_MAX_DISCOUNT RBPF_REAL(0.12)     /* Max 12% reduction */

/* Bounds */
#define DEFAULT_LAMBDA_FLOOR RBPF_REAL(0.980)    /* N_eff ≈ 50 minimum */
#define DEFAULT_LAMBDA_CEILING RBPF_REAL(0.9995) /* N_eff ≈ 2000 maximum */

/* EMA parameters */
#define DEFAULT_BASELINE_ALPHA RBPF_REAL(0.01) /* Very slow (τ ≈ 100) - slow to adapt */
#define DEFAULT_SIGNAL_ALPHA RBPF_REAL(0.15)   /* Faster reaction */

/* Cooldown */
#define DEFAULT_COOLDOWN_TICKS 10

/* Intervention threshold (z-score above which we count as intervention) */
#define INTERVENTION_THRESHOLD RBPF_REAL(1.5)

/*============================================================================
 * INITIALIZATION
 *============================================================================*/

/**
 * Initialize adaptive forgetting to defaults
 *
 * Called internally by rbpf_ext_create(). Sets up default values
 * but leaves disabled until explicitly enabled.
 */
void rbpf_adaptive_forgetting_init(RBPF_AdaptiveForgetting *af)
{
    if (!af)
        return;

    memset(af, 0, sizeof(*af));

    af->enabled = 0;
    af->signal_source = ADAPT_SIGNAL_COMBINED; /* Recommended default */

    /* Regime baselines */
    for (int r = 0; r < RBPF_MAX_REGIMES; r++)
    {
        af->lambda_per_regime[r] = DEFAULT_LAMBDA_REGIME[r];
    }

    /* Surprise tracking - initialize to reasonable "calm market" values */
    af->surprise_baseline = RBPF_REAL(5.0); /* Typical -log(p) in normal conditions */
    af->surprise_var = RBPF_REAL(1.0);      /* Unit variance initially */
    af->surprise_ema_alpha = DEFAULT_BASELINE_ALPHA;

    af->signal_ema = RBPF_REAL(0.0);
    af->signal_ema_alpha = DEFAULT_SIGNAL_ALPHA;

    /* Sigmoid parameters */
    af->sigmoid_center = DEFAULT_SIGMOID_CENTER;
    af->sigmoid_steepness = DEFAULT_SIGMOID_STEEPNESS;
    af->max_discount = DEFAULT_MAX_DISCOUNT;

    /* Bounds */
    af->lambda_floor = DEFAULT_LAMBDA_FLOOR;
    af->lambda_ceiling = DEFAULT_LAMBDA_CEILING;

    /* Cooldown */
    af->cooldown_ticks = DEFAULT_COOLDOWN_TICKS;
    af->cooldown_remaining = 0;

    /* Output */
    af->lambda_current = RBPF_REAL(0.998); /* Safe default */
    af->surprise_current = RBPF_REAL(0.0);
    af->surprise_zscore = RBPF_REAL(0.0);
    af->discount_applied = RBPF_REAL(0.0);

    /* Statistics */
    af->interventions = 0;
    af->max_surprise_seen = RBPF_REAL(0.0);
}

/*============================================================================
 * CORE UPDATE FUNCTION
 *============================================================================*/

/**
 * Sigmoid function for continuous intervention
 *
 * Maps z-score to discount amount [0, max_discount]
 */
static inline rbpf_real_t sigmoid_discount(
    const RBPF_AdaptiveForgetting *af,
    rbpf_real_t z)
{
    if (z < RBPF_REAL(-5.0))
        return RBPF_REAL(0.0); /* Avoid exp overflow */
    if (z > RBPF_REAL(10.0))
        return af->max_discount; /* Saturate */

    rbpf_real_t x = -af->sigmoid_steepness * (z - af->sigmoid_center);
    return af->max_discount / (RBPF_REAL(1.0) + rbpf_exp(x));
}

/**
 * Main update function - call AFTER rbpf_ksc_update()
 *
 * Uses the marginal likelihood from the update step to compute surprise
 * and adjust the forgetting factor.
 *
 * SEPARATION OF CONCERNS:
 *   PATH A: Predictive Surprise (Drift Detection)
 *     - Uses Z-score normalization (is today worse than yesterday?)
 *     - Triggers on gradual regime shifts
 *
 *   PATH B: Structural Surprise (Shock Detection)
 *     - Uses ABSOLUTE outlier fraction (no history needed)
 *     - Triggers immediately on heavy outlier usage
 *     - Bypasses Z-score entirely
 *
 * @param ext              Extended RBPF handle
 * @param marginal_lik     Marginal likelihood from rbpf_ksc_update()
 * @param dominant_regime  Current dominant regime (for baseline λ)
 */
void rbpf_adaptive_forgetting_update(
    RBPF_Extended *ext,
    rbpf_real_t marginal_lik,
    int dominant_regime)
{
    if (!ext)
        return;

    RBPF_AdaptiveForgetting *af = &ext->adaptive_forgetting;

    if (!af->enabled)
    {
        /* When disabled, use fixed λ from Storvik config */
        return;
    }

    /*═══════════════════════════════════════════════════════════════════════
     * PATH A: PREDICTIVE SURPRISE (Drift Detection)
     *
     * Uses Z-score to detect when data fits worse than usual.
     * Good for: Gradual regime shifts, persistent misspecification
     *═══════════════════════════════════════════════════════════════════════*/

    /* Compute predictive surprise */
    if (marginal_lik < RBPF_REAL(1e-30))
    {
        marginal_lik = RBPF_REAL(1e-30);
    }
    rbpf_real_t pred_surprise = -rbpf_log(marginal_lik);

    af->surprise_current = pred_surprise;

    /* Track maximum for diagnostics */
    if (pred_surprise > af->max_surprise_seen)
    {
        af->max_surprise_seen = pred_surprise;
    }

    /* Update baseline statistics (slow EMA, only when not intervening) */
    rbpf_real_t alpha_base = af->surprise_ema_alpha;

    if (af->cooldown_remaining == 0 && af->surprise_zscore < RBPF_REAL(1.0))
    {
        af->surprise_baseline = alpha_base * pred_surprise +
                                (RBPF_REAL(1.0) - alpha_base) * af->surprise_baseline;

        rbpf_real_t delta = pred_surprise - af->surprise_baseline;
        af->surprise_var = alpha_base * (delta * delta) +
                           (RBPF_REAL(1.0) - alpha_base) * af->surprise_var;

        if (af->surprise_var < RBPF_REAL(0.01))
        {
            af->surprise_var = RBPF_REAL(0.01);
        }
    }

    /* Compute Z-score */
    rbpf_real_t surprise_std = rbpf_sqrt(af->surprise_var);
    af->surprise_zscore = (pred_surprise - af->surprise_baseline) / (surprise_std + RBPF_REAL(1e-6));

    /* Smooth the Z-score signal */
    af->signal_ema = af->signal_ema_alpha * af->surprise_zscore +
                     (RBPF_REAL(1.0) - af->signal_ema_alpha) * af->signal_ema;

    /* Compute discount from predictive surprise */
    rbpf_real_t discount_pred = sigmoid_discount(af, af->signal_ema);

    /*═══════════════════════════════════════════════════════════════════════
     * PATH B: STRUCTURAL SURPRISE (Shock Detection)
     *
     * Direct penalty based on outlier fraction. NO HISTORY NEEDED.
     * This is ABSOLUTE - 80% outliers is always bad, period.
     *
     * Bypasses Z-score entirely to avoid the "Baseline Trap":
     *   - Z-score normalizes everything relative to history
     *   - But outlier fraction has an absolute meaning
     *   - 80% outliers = crisis, even if yesterday was also 80%
     *═══════════════════════════════════════════════════════════════════════*/

    rbpf_real_t out_frac = ext->last_outlier_fraction;
    rbpf_real_t discount_struct = RBPF_REAL(0.0);

    /* Threshold-based activation:
     *   <20% outliers: Normal market noise, no penalty
     *   20-80% outliers: Linear ramp from 0 to max_discount
     *   >80% outliers: Full max_discount
     */
    if (out_frac > RBPF_REAL(0.20))
    {
        rbpf_real_t intensity = (out_frac - RBPF_REAL(0.20)) / RBPF_REAL(0.60);
        if (intensity > RBPF_REAL(1.0))
            intensity = RBPF_REAL(1.0);
        discount_struct = af->max_discount * intensity;
    }

    /*═══════════════════════════════════════════════════════════════════════
     * COMBINE: Take the STRONGER penalty
     *
     * MAX ensures we adapt if EITHER:
     *   - Data fits poorly (drift) → discount_pred triggers
     *   - Data fits only via outliers (shock) → discount_struct triggers
     *═══════════════════════════════════════════════════════════════════════*/

    rbpf_real_t final_discount;

    switch (af->signal_source)
    {
    case ADAPT_SIGNAL_REGIME:
        /* Pure regime baseline, no surprise modulation */
        final_discount = RBPF_REAL(0.0);
        break;

    case ADAPT_SIGNAL_OUTLIER_FRAC:
        /* Only structural (for testing) */
        final_discount = discount_struct;
        break;

    case ADAPT_SIGNAL_PREDICTIVE_SURPRISE:
        /* Only predictive (for testing) */
        final_discount = discount_pred;
        break;

    case ADAPT_SIGNAL_COMBINED:
    default:
        /* RECOMMENDED: Max of both paths */
        final_discount = (discount_pred > discount_struct) ? discount_pred : discount_struct;
        break;
    }

    af->discount_applied = final_discount;

    /*═══════════════════════════════════════════════════════════════════════
     * COMPUTE LAMBDA
     *═══════════════════════════════════════════════════════════════════════*/

    rbpf_real_t base_lambda = af->lambda_per_regime[dominant_regime];
    rbpf_real_t lambda = base_lambda * (RBPF_REAL(1.0) - final_discount);

    /*═══════════════════════════════════════════════════════════════════════
     * COOLDOWN: Hold low λ after intervention
     *
     * Prevents oscillation between high/low λ.
     *═══════════════════════════════════════════════════════════════════════*/

    if (af->cooldown_remaining > 0)
    {
        /* Keep previous (low) λ during cooldown */
        lambda = af->lambda_current;
        af->cooldown_remaining--;
    }
    else if (final_discount > RBPF_REAL(0.05) || af->surprise_zscore > INTERVENTION_THRESHOLD)
    {
        /* Start cooldown on significant intervention */
        af->cooldown_remaining = af->cooldown_ticks;
        af->interventions++;
    }

    /*═══════════════════════════════════════════════════════════════════════
     * BOUNDS AND OUTPUT
     *═══════════════════════════════════════════════════════════════════════*/

    if (lambda < af->lambda_floor)
        lambda = af->lambda_floor;
    if (lambda > af->lambda_ceiling)
        lambda = af->lambda_ceiling;

    af->lambda_current = lambda;

    /*═══════════════════════════════════════════════════════════════════════
     * PUSH TO STORVIK
     *═══════════════════════════════════════════════════════════════════════*/

    if (ext->storvik_initialized)
    {
        param_learn_set_forgetting(&ext->storvik, 1, lambda);
    }
}

/*============================================================================
 * API FUNCTIONS
 *============================================================================*/

void rbpf_ext_enable_adaptive_forgetting(RBPF_Extended *ext)
{
    if (!ext)
        return;

    RBPF_AdaptiveForgetting *af = &ext->adaptive_forgetting;

    /* Initialize to defaults if not already */
    rbpf_adaptive_forgetting_init(af);

    af->enabled = 1;
    af->signal_source = ADAPT_SIGNAL_COMBINED;

    /* Also enable Storvik forgetting as the underlying mechanism */
    if (ext->storvik_initialized)
    {
        param_learn_set_forgetting(&ext->storvik, 1, af->lambda_per_regime[1]); /* Start at R1 */
    }
}

void rbpf_ext_enable_adaptive_forgetting_mode(RBPF_Extended *ext, RBPF_AdaptSignal signal)
{
    if (!ext)
        return;

    rbpf_ext_enable_adaptive_forgetting(ext);
    ext->adaptive_forgetting.signal_source = signal;
}

void rbpf_ext_disable_adaptive_forgetting(RBPF_Extended *ext)
{
    if (!ext)
        return;
    ext->adaptive_forgetting.enabled = 0;
}

void rbpf_ext_set_regime_lambda(RBPF_Extended *ext, int regime, rbpf_real_t lambda)
{
    if (!ext || regime < 0 || regime >= RBPF_MAX_REGIMES)
        return;

    /* Clamp to reasonable range */
    if (lambda < RBPF_REAL(0.9))
        lambda = RBPF_REAL(0.9);
    if (lambda > RBPF_REAL(0.9999))
        lambda = RBPF_REAL(0.9999);

    ext->adaptive_forgetting.lambda_per_regime[regime] = lambda;
}

void rbpf_ext_set_adaptive_sigmoid(RBPF_Extended *ext,
                                   rbpf_real_t center,
                                   rbpf_real_t steepness,
                                   rbpf_real_t max_discount)
{
    if (!ext)
        return;

    RBPF_AdaptiveForgetting *af = &ext->adaptive_forgetting;

    af->sigmoid_center = center;
    af->sigmoid_steepness = steepness;

    /* Clamp max_discount to prevent excessive forgetting */
    if (max_discount > RBPF_REAL(0.5))
        max_discount = RBPF_REAL(0.5);
    if (max_discount < RBPF_REAL(0.01))
        max_discount = RBPF_REAL(0.01);
    af->max_discount = max_discount;
}

void rbpf_ext_set_adaptive_bounds(RBPF_Extended *ext,
                                  rbpf_real_t floor,
                                  rbpf_real_t ceiling)
{
    if (!ext)
        return;

    /* Ensure floor < ceiling and both in valid range */
    if (floor < RBPF_REAL(0.8))
        floor = RBPF_REAL(0.8);
    if (ceiling > RBPF_REAL(0.9999))
        ceiling = RBPF_REAL(0.9999);
    if (floor >= ceiling)
        floor = ceiling - RBPF_REAL(0.01);

    ext->adaptive_forgetting.lambda_floor = floor;
    ext->adaptive_forgetting.lambda_ceiling = ceiling;
}

void rbpf_ext_set_adaptive_smoothing(RBPF_Extended *ext,
                                     rbpf_real_t baseline_alpha,
                                     rbpf_real_t signal_alpha)
{
    if (!ext)
        return;

    /* Clamp to reasonable range (0.001, 0.5) */
    if (baseline_alpha < RBPF_REAL(0.001))
        baseline_alpha = RBPF_REAL(0.001);
    if (baseline_alpha > RBPF_REAL(0.5))
        baseline_alpha = RBPF_REAL(0.5);
    if (signal_alpha < RBPF_REAL(0.001))
        signal_alpha = RBPF_REAL(0.001);
    if (signal_alpha > RBPF_REAL(0.5))
        signal_alpha = RBPF_REAL(0.5);

    ext->adaptive_forgetting.surprise_ema_alpha = baseline_alpha;
    ext->adaptive_forgetting.signal_ema_alpha = signal_alpha;
}

void rbpf_ext_set_adaptive_cooldown(RBPF_Extended *ext, int ticks)
{
    if (!ext)
        return;

    if (ticks < 0)
        ticks = 0;
    if (ticks > 100)
        ticks = 100;

    ext->adaptive_forgetting.cooldown_ticks = ticks;
}

rbpf_real_t rbpf_ext_get_current_lambda(const RBPF_Extended *ext)
{
    if (!ext)
        return RBPF_REAL(0.998);
    return ext->adaptive_forgetting.lambda_current;
}

rbpf_real_t rbpf_ext_get_surprise_zscore(const RBPF_Extended *ext)
{
    if (!ext)
        return RBPF_REAL(0.0);
    return ext->adaptive_forgetting.surprise_zscore;
}

void rbpf_ext_get_adaptive_stats(const RBPF_Extended *ext,
                                 uint64_t *interventions,
                                 rbpf_real_t *current_lambda,
                                 rbpf_real_t *max_surprise)
{
    if (!ext)
        return;

    const RBPF_AdaptiveForgetting *af = &ext->adaptive_forgetting;

    if (interventions)
        *interventions = af->interventions;
    if (current_lambda)
        *current_lambda = af->lambda_current;
    if (max_surprise)
        *max_surprise = af->max_surprise_seen;
}

/*============================================================================
 * DIAGNOSTICS
 *============================================================================*/

void rbpf_ext_print_adaptive_config(const RBPF_Extended *ext)
{
    if (!ext)
        return;

    const RBPF_AdaptiveForgetting *af = &ext->adaptive_forgetting;

    static const char *signal_names[] = {
        "REGIME", "OUTLIER_FRAC", "PREDICTIVE_SURPRISE", "COMBINED"};

    printf("\n┌─────────────────────────────────────────────────────────────┐\n");
    printf("│             Adaptive Forgetting Configuration               │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");

    if (!af->enabled)
    {
        printf("│  Status: DISABLED                                           │\n");
        printf("└─────────────────────────────────────────────────────────────┘\n");
        return;
    }

    printf("│  Status:        ENABLED                                     │\n");
    printf("│  Signal Source: %-20s                      │\n", signal_names[af->signal_source]);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Regime Baselines (λ → N_eff):                              │\n");
    for (int r = 0; r < 4; r++)
    {
        rbpf_real_t lam = af->lambda_per_regime[r];
        int n_eff = (int)(RBPF_REAL(1.0) / (RBPF_REAL(1.0) - lam));
        printf("│    R%d: λ=%.4f → N_eff≈%4d                                │\n",
               r, (float)lam, n_eff);
    }
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Sigmoid Response:                                          │\n");
    printf("│    Center (z):     %.2f                                     │\n", (float)af->sigmoid_center);
    printf("│    Steepness:      %.2f                                     │\n", (float)af->sigmoid_steepness);
    printf("│    Max Discount:   %.1f%%                                    │\n", (float)af->max_discount * 100);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Bounds:                                                    │\n");
    printf("│    Floor:   λ=%.3f (N_eff≈%d)                             │\n",
           (float)af->lambda_floor, (int)(1.0f / (1.0f - af->lambda_floor)));
    printf("│    Ceiling: λ=%.4f (N_eff≈%d)                            │\n",
           (float)af->lambda_ceiling, (int)(1.0f / (1.0f - af->lambda_ceiling)));
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Smoothing:                                                 │\n");
    printf("│    Baseline α: %.3f (τ≈%.0f ticks)                         │\n",
           (float)af->surprise_ema_alpha, 1.0f / (float)af->surprise_ema_alpha);
    printf("│    Signal α:   %.3f (τ≈%.0f ticks)                         │\n",
           (float)af->signal_ema_alpha, 1.0f / (float)af->signal_ema_alpha);
    printf("│    Cooldown:   %d ticks                                     │\n", af->cooldown_ticks);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Current State:                                             │\n");
    printf("│    λ_current:      %.4f                                    │\n", (float)af->lambda_current);
    printf("│    Surprise z:     %+.2f                                    │\n", (float)af->surprise_zscore);
    printf("│    Discount:       %.1f%%                                    │\n", (float)af->discount_applied * 100);
    printf("│    Cooldown left:  %d                                       │\n", af->cooldown_remaining);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Statistics:                                                │\n");
    printf("│    Interventions:  %llu                                     │\n", (unsigned long long)af->interventions);
    printf("│    Max Surprise:   %.2f                                    │\n", (float)af->max_surprise_seen);
    printf("│    Baseline μ:     %.2f                                    │\n", (float)af->surprise_baseline);
    printf("│    Baseline σ:     %.2f                                    │\n", (float)rbpf_sqrt(af->surprise_var));
    printf("└─────────────────────────────────────────────────────────────┘\n");
}