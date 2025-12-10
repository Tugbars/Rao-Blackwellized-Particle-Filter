/**
 * @file hawkes_intensity.h
 * @brief Hawkes Self-Exciting Point Process for RBPF
 *
 * Replaces Omori power-law decay with proper self-exciting process.
 * Key improvement: events trigger future events (branching), not just decay.
 *
 * Model:
 *   λ(t) = μ + Σᵢ α·g(t - tᵢ)·h(mᵢ)
 *
 * Where:
 *   μ     = baseline intensity
 *   α     = excitation strength  
 *   g(·)  = exponential kernel: exp(-β·Δt)
 *   h(·)  = mark function: magnitude impact
 *   tᵢ,mᵢ = past event times and magnitudes
 *
 * For RBPF integration:
 *   - High intensity → boost upward regime transitions
 *   - Per-regime parameters (crisis has different dynamics)
 *   - Marks = |return| or squared return
 *
 * Reference: Bacry, Mastromatteo, Muzy (2015) "Hawkes Processes in Finance"
 */

#ifndef HAWKES_INTENSITY_H
#define HAWKES_INTENSITY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

#define HAWKES_MAX_EVENTS 128      /* Circular buffer size */
#define HAWKES_MAX_REGIMES 8

/**
 * Kernel type for decay function g(Δt)
 */
typedef enum {
    HAWKES_KERNEL_EXPONENTIAL,     /* g(t) = exp(-β·t) - standard */
    HAWKES_KERNEL_POWER_LAW,       /* g(t) = (1 + t/c)^(-p) - Omori-like */
    HAWKES_KERNEL_SUM_EXP,         /* g(t) = Σ wᵢ·exp(-βᵢ·t) - multi-scale */
} HawkesKernelType;

/**
 * Mark impact function h(m)
 */
typedef enum {
    HAWKES_MARK_NONE,              /* h(m) = 1 (unmarked) */
    HAWKES_MARK_LINEAR,            /* h(m) = m */
    HAWKES_MARK_QUADRATIC,         /* h(m) = m² */
    HAWKES_MARK_EXPONENTIAL,       /* h(m) = exp(γ·m) - 1 */
} HawkesMarkType;

/**
 * Per-regime Hawkes parameters
 */
typedef struct {
    float mu;                      /* Baseline intensity */
    float alpha;                   /* Excitation strength (branching ratio = α/β) */
    float beta;                    /* Decay rate (half-life ≈ 0.693/β) */
    float threshold;               /* |return| threshold to trigger event */
    
    /* For multi-scale kernel */
    float beta_fast;               /* Fast decay component */
    float beta_slow;               /* Slow decay component */
    float weight_fast;             /* Weight on fast (slow = 1 - fast) */
    
    /* For mark impact */
    float gamma;                   /* Mark sensitivity */
    
} HawkesRegimeParams;

/**
 * Main configuration
 */
typedef struct {
    int n_regimes;
    HawkesKernelType kernel;
    HawkesMarkType mark_type;
    
    HawkesRegimeParams regime[HAWKES_MAX_REGIMES];
    
    /* Regime transition modification */
    int modify_transitions;        /* 1 = use intensity to modify P(regime) */
    float intensity_threshold;     /* Above this, start modifying */
    float max_transition_boost;    /* Maximum boost to upward transition */
    
    /* Numerical stability */
    float min_intensity;           /* Floor for λ(t) */
    float max_intensity;           /* Ceiling for λ(t) */
    
} HawkesConfig;

/*═══════════════════════════════════════════════════════════════════════════
 * STATE
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Single event record
 */
typedef struct {
    float time;                    /* Event time (tick index or continuous) */
    float mark;                    /* Event magnitude (|return|, variance, etc.) */
    int   regime;                  /* Regime when event occurred */
} HawkesEvent;

/**
 * Hawkes process state
 */
typedef struct {
    HawkesConfig config;
    
    /* Event history (circular buffer) */
    HawkesEvent events[HAWKES_MAX_EVENTS];
    int head;                      /* Next write position */
    int count;                     /* Number of events stored */
    
    /* Current state */
    float current_time;
    float intensity;               /* Current λ(t) */
    float intensity_baseline;      /* μ for current regime */
    float intensity_excited;       /* Excitation component only */
    int   current_regime;
    
    /* Running statistics */
    float intensity_ema;           /* Smoothed intensity for diagnostics */
    float event_rate_ema;          /* Smoothed event rate */
    int   total_events;
    
    /* Cached computation */
    float sum_kernels;             /* Σ g(t - tᵢ)·h(mᵢ) */
    int   cache_valid;
    
} HawkesState;

/*═══════════════════════════════════════════════════════════════════════════
 * API - LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Get default configuration for financial volatility
 */
HawkesConfig hawkes_config_defaults(void);

/**
 * Get configuration tuned for crisis detection
 */
HawkesConfig hawkes_config_crisis_sensitive(void);

/**
 * Initialize Hawkes state
 */
int hawkes_init(HawkesState *state, const HawkesConfig *config);

/**
 * Reset state (keep config)
 */
void hawkes_reset(HawkesState *state);

/**
 * Free resources (currently no-op, but future-proof)
 */
void hawkes_free(HawkesState *state);

/*═══════════════════════════════════════════════════════════════════════════
 * API - CORE UPDATE
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Update Hawkes intensity after observing a return
 *
 * @param state     Hawkes state
 * @param time      Current time (tick index)
 * @param obs_return Observed return (raw, not absolute)
 * @param regime    Current regime estimate from RBPF
 * @return Updated intensity λ(t)
 */
float hawkes_update(HawkesState *state, float time, float obs_return, int regime);

/**
 * Get current intensity without updating (query only)
 */
float hawkes_get_intensity(const HawkesState *state);

/**
 * Get intensity at specific future time (for prediction)
 */
float hawkes_predict_intensity(const HawkesState *state, float future_time);

/*═══════════════════════════════════════════════════════════════════════════
 * API - RBPF INTEGRATION
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Modify regime transition probabilities based on Hawkes intensity
 *
 * High intensity → boost probability of transitioning to higher (more volatile) regime
 * Low intensity → boost probability of transitioning to lower (calmer) regime
 *
 * @param state         Hawkes state
 * @param trans_row     Transition probabilities from current regime [n_regimes]
 * @param current_regime Current regime index
 * @param n_regimes     Number of regimes
 *
 * Modifies trans_row in place. Row will still sum to 1.
 */
void hawkes_modify_transition_probs(const HawkesState *state,
                                    float *trans_row,
                                    int current_regime,
                                    int n_regimes);

/**
 * Get likelihood contribution from Hawkes process
 *
 * For particle weighting: particles in regimes consistent with
 * current intensity should get higher weight.
 *
 * @param state     Hawkes state  
 * @param regime    Proposed regime
 * @param vol       Proposed volatility
 * @return Log-likelihood contribution
 */
float hawkes_regime_loglik(const HawkesState *state, int regime, float vol);

/**
 * Get expected volatility given current intensity
 * Maps intensity to volatility scale for comparison with RBPF output
 */
float hawkes_expected_vol(const HawkesState *state, int regime);

/*═══════════════════════════════════════════════════════════════════════════
 * API - CALIBRATION
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Estimate Hawkes parameters from historical data
 *
 * Uses MLE with EM algorithm.
 *
 * @param returns       Array of returns [n]
 * @param n             Number of observations
 * @param threshold     |return| threshold for events
 * @param out_mu        Output: estimated baseline
 * @param out_alpha     Output: estimated excitation
 * @param out_beta      Output: estimated decay
 * @return 0 on success, -1 on convergence failure
 */
int hawkes_calibrate_mle(const float *returns, int n, float threshold,
                         float *out_mu, float *out_alpha, float *out_beta);

/**
 * Quick method-of-moments calibration
 * Less accurate but much faster, good for initial guess
 */
void hawkes_calibrate_moments(const float *returns, int n, float threshold,
                              float *out_mu, float *out_alpha, float *out_beta);

/*═══════════════════════════════════════════════════════════════════════════
 * API - DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Compute branching ratio n = α/β
 * 
 * n < 1: subcritical (stationary)
 * n = 1: critical (non-stationary)
 * n > 1: supercritical (explosive) - invalid for stationary model
 */
float hawkes_branching_ratio(const HawkesState *state, int regime);

/**
 * Compute half-life of excitation decay
 */
float hawkes_half_life(const HawkesState *state, int regime);

/**
 * Print current state for debugging
 */
void hawkes_print_state(const HawkesState *state);

/**
 * Print configuration
 */
void hawkes_print_config(const HawkesConfig *config);

/*═══════════════════════════════════════════════════════════════════════════
 * API - BATCH PROCESSING
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Process multiple returns at once
 * More efficient than calling hawkes_update in loop
 *
 * @param state     Hawkes state
 * @param returns   Array of returns [n]
 * @param regimes   Array of regimes [n] (can be NULL to use state's regime)
 * @param n         Number of observations
 * @param out_intensity Output intensity array [n] (can be NULL)
 */
void hawkes_update_batch(HawkesState *state,
                         const float *returns,
                         const int *regimes,
                         int n,
                         float *out_intensity);

#ifdef __cplusplus
}
#endif

#endif /* HAWKES_INTENSITY_H */
