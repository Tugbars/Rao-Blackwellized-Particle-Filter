/**
 * @file rbpf_watchdog.h
 * @brief Divergence Watchdog ("Defibrillator") for RBPF
 *
 * Monitors filter health and triggers controlled re-initialization when
 * the particle filter diverges. This is the critical safety rail that
 * converts a research prototype into a production-grade system.
 *
 * Based on: Thrun et al. (2005) - Probabilistic Robotics
 *           "Kidnapped Robot Problem" recovery strategies
 *
 * FAILURE MODES DETECTED:
 *   1. ESS Collapse      - All weight concentrated on few particles
 *   2. NaN Weights       - Numerical underflow/overflow
 *   3. Inf Weights       - Likelihood explosion
 *   4. Likelihood Floor  - Model completely misspecified
 *   5. State Divergence  - Estimate wildly inconsistent with observation
 *   6. Regime Deadlock   - Stuck in wrong regime despite evidence
 *
 * RESET BEHAVIOR:
 *   - NOT a crash or blind restart
 *   - Controlled re-initialization around best available estimate
 *   - Preserves learned parameters (optional)
 *   - Logs event for post-mortem analysis
 *   - Signals upstream systems (risk manager, execution)
 *
 * USAGE:
 *   // In rbpf_ext_step():
 *   RBPF_ResetReason reason = rbpf_watchdog_check(ext);
 *   if (reason != RESET_NONE) {
 *       rbpf_watchdog_reset(ext, reason);
 *   }
 */

#ifndef RBPF_WATCHDOG_H
#define RBPF_WATCHDOG_H

#include <stdint.h>

/* Forward declaration */
typedef struct RBPF_Extended RBPF_Extended;

/*═══════════════════════════════════════════════════════════════════════════
 * RESET REASON CODES
 *═══════════════════════════════════════════════════════════════════════════*/

typedef enum
{
    RESET_NONE = 0, /* Filter healthy, no reset needed */

    /* Weight-based failures */
    RESET_ESS_COLLAPSE, /* ESS < critical threshold */
    RESET_NAN_WEIGHT,   /* NaN detected in weights */
    RESET_INF_WEIGHT,   /* Inf detected in weights */
    RESET_ZERO_WEIGHTS, /* All weights effectively zero */

    /* Likelihood-based failures */
    RESET_LIK_COLLAPSE, /* Marginal likelihood < floor */
    RESET_LIK_NAN,      /* Likelihood computation failed */

    /* State-based failures */
    RESET_STATE_DIVERGENCE, /* State estimate wildly off */
    RESET_STATE_NAN,        /* NaN in state estimates */
    RESET_STATE_EXPLOSION,  /* |h_t| unreasonably large */

    /* Regime-based failures */
    RESET_REGIME_DEADLOCK, /* Stuck in wrong regime */

    /* Manual trigger */
    RESET_MANUAL, /* Explicitly requested by user */

    RESET_REASON_COUNT /* Sentinel for iteration */
} RBPF_ResetReason;

/*═══════════════════════════════════════════════════════════════════════════
 * RESET SEVERITY LEVELS
 *═══════════════════════════════════════════════════════════════════════════*/

typedef enum
{
    SEVERITY_NONE = 0,
    SEVERITY_SOFT,    /* Re-diversify particles only */
    SEVERITY_MEDIUM,  /* Reset state, keep parameters */
    SEVERITY_HARD,    /* Full reset including parameters */
    SEVERITY_CRITICAL /* Emergency: signal upstream, halt trading */
} RBPF_ResetSeverity;

/*═══════════════════════════════════════════════════════════════════════════
 * WATCHDOG CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    int enabled;

    /*───────────────────────────────────────────────────────────────────────
     * ESS Thresholds
     * ESS = 1/Σ(w_i²), ranges from 1 (degenerate) to N (uniform)
     *─────────────────────────────────────────────────────────────────────*/
    float ess_critical_ratio; /* Trigger if ESS < ratio * N (default: 0.05) */
    float ess_warning_ratio;  /* Log warning if ESS < ratio * N (default: 0.15) */

    /*───────────────────────────────────────────────────────────────────────
     * Likelihood Thresholds
     *─────────────────────────────────────────────────────────────────────*/
    float log_lik_floor;   /* Trigger if log_lik < floor (default: -1e6) */
    float log_lik_warning; /* Log warning if log_lik < warning (default: -1e4) */

    /*───────────────────────────────────────────────────────────────────────
     * State Divergence Thresholds
     * Compare estimated vol to observation-implied vol
     *─────────────────────────────────────────────────────────────────────*/
    float state_divergence_sigma; /* Trigger if divergence > N sigma (default: 8.0) */
    float state_max_abs;          /* Trigger if |h_t| > max (default: 10.0, ~22000% vol) */

    /*───────────────────────────────────────────────────────────────────────
     * Regime Deadlock Detection
     * Trigger if we're in low regime but seeing high-vol observations
     *─────────────────────────────────────────────────────────────────────*/
    int regime_deadlock_window;   /* Ticks to check (default: 10) */
    float regime_deadlock_thresh; /* Avg |return| threshold (default: 0.05) */

    /*───────────────────────────────────────────────────────────────────────
     * Cooldown & Rate Limiting
     *─────────────────────────────────────────────────────────────────────*/
    int min_ticks_between_resets; /* Prevent reset storms (default: 50) */
    int max_resets_per_session;   /* Circuit breaker (default: 10, 0 = unlimited) */

    /*───────────────────────────────────────────────────────────────────────
     * Reset Behavior Configuration
     *─────────────────────────────────────────────────────────────────────*/
    int preserve_learned_params; /* Keep Storvik stats on reset? (default: 1) */
    float reset_variance_scale;  /* Multiply init variance by this (default: 2.0) */
    int clear_hawkes_on_reset;   /* Reset Hawkes intensity? (default: 1) */

} RBPF_WatchdogConfig;

/*═══════════════════════════════════════════════════════════════════════════
 * WATCHDOG STATE
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    RBPF_WatchdogConfig config;

    /*───────────────────────────────────────────────────────────────────────
     * Runtime State
     *─────────────────────────────────────────────────────────────────────*/
    int ticks_since_last_reset;
    int total_resets;
    int session_resets;          /* Resets this session (for circuit breaker) */
    int circuit_breaker_tripped; /* 1 if max resets exceeded */

    /*───────────────────────────────────────────────────────────────────────
     * Last Reset Info (for diagnostics)
     *─────────────────────────────────────────────────────────────────────*/
    RBPF_ResetReason last_reset_reason;
    RBPF_ResetSeverity last_reset_severity;
    uint64_t last_reset_tick;
    float last_reset_ess;
    float last_reset_log_lik;
    float last_reset_state_mean;

    /*───────────────────────────────────────────────────────────────────────
     * Rolling Diagnostics (for deadlock detection)
     *─────────────────────────────────────────────────────────────────────*/
    float recent_returns[32]; /* Circular buffer */
    int recent_returns_idx;
    int recent_regimes[32]; /* Circular buffer */
    int recent_regimes_idx;

    /*───────────────────────────────────────────────────────────────────────
     * Warning Counters (for logging rate limiting)
     *─────────────────────────────────────────────────────────────────────*/
    int consecutive_ess_warnings;
    int consecutive_lik_warnings;
    int consecutive_divergence_count; /* For state divergence detection */

    /*───────────────────────────────────────────────────────────────────────
     * Callback for upstream notification (optional)
     *─────────────────────────────────────────────────────────────────────*/
    void (*on_reset_callback)(RBPF_ResetReason reason, RBPF_ResetSeverity severity, void *user_data);
    void *callback_user_data;

} RBPF_Watchdog;

/*═══════════════════════════════════════════════════════════════════════════
 * RESET EVENT STRUCTURE (for logging/replay)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    uint64_t tick;
    RBPF_ResetReason reason;
    RBPF_ResetSeverity severity;

    /* State before reset */
    float ess_before;
    float log_lik_before;
    float state_mean_before;
    float state_var_before;
    int regime_before;
    float hawkes_intensity_before;

    /* Trigger values */
    float trigger_value;     /* The value that triggered reset */
    float trigger_threshold; /* The threshold it exceeded */

} RBPF_ResetEvent;

/*═══════════════════════════════════════════════════════════════════════════
 * API FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Initialize watchdog with default configuration
 */
void rbpf_watchdog_init(RBPF_Watchdog *wd);

/**
 * Initialize watchdog with custom configuration
 */
void rbpf_watchdog_init_config(RBPF_Watchdog *wd, const RBPF_WatchdogConfig *config);

/**
 * Set default configuration values
 */
void rbpf_watchdog_default_config(RBPF_WatchdogConfig *config);

/**
 * Check filter health
 *
 * Call this after each rbpf_ksc_update() but before using outputs.
 *
 * @param ext      Extended RBPF handle
 * @param wd       Watchdog state
 * @param obs      Current observation (for deadlock detection)
 * @return         RESET_NONE if healthy, otherwise the failure reason
 */
RBPF_ResetReason rbpf_watchdog_check(RBPF_Extended *ext, RBPF_Watchdog *wd, float obs);

/**
 * Determine severity for a given reset reason
 */
RBPF_ResetSeverity rbpf_watchdog_get_severity(RBPF_ResetReason reason);

/**
 * Perform controlled reset
 *
 * This is NOT a crash. It's a controlled re-initialization that:
 *   1. Logs the event (reason, state before reset)
 *   2. Re-initializes particles around best available estimate
 *   3. Optionally preserves learned parameters
 *   4. Signals upstream via callback (if registered)
 *
 * @param ext      Extended RBPF handle
 * @param wd       Watchdog state
 * @param reason   Why we're resetting
 * @param event    Optional: filled with reset details (can be NULL)
 * @return         0 on success, -1 if circuit breaker tripped
 */
int rbpf_watchdog_reset(RBPF_Extended *ext, RBPF_Watchdog *wd,
                        RBPF_ResetReason reason, RBPF_ResetEvent *event);

/**
 * Manual reset trigger
 *
 * For use by external systems (e.g., risk manager says "reset now")
 */
int rbpf_watchdog_manual_reset(RBPF_Extended *ext, RBPF_Watchdog *wd);

/**
 * Register callback for reset notifications
 *
 * Callback is invoked AFTER reset completes.
 * Use for: logging, alerting, pausing trading, etc.
 */
void rbpf_watchdog_set_callback(RBPF_Watchdog *wd,
                                void (*callback)(RBPF_ResetReason, RBPF_ResetSeverity, void *),
                                void *user_data);

/**
 * Reset session counters (call at start of trading day)
 */
void rbpf_watchdog_new_session(RBPF_Watchdog *wd);

/**
 * Check if circuit breaker is tripped
 */
int rbpf_watchdog_is_circuit_breaker_tripped(const RBPF_Watchdog *wd);

/**
 * Get human-readable reason string
 */
const char *rbpf_watchdog_reason_str(RBPF_ResetReason reason);

/**
 * Get human-readable severity string
 */
const char *rbpf_watchdog_severity_str(RBPF_ResetSeverity severity);

/**
 * Print watchdog status (for debugging)
 */
void rbpf_watchdog_print_status(const RBPF_Watchdog *wd);

/**
 * Print reset event (for logging)
 */
void rbpf_watchdog_print_event(const RBPF_ResetEvent *event);

/*═══════════════════════════════════════════════════════════════════════════
 * CONVENIENCE MACROS
 *═══════════════════════════════════════════════════════════════════════════*/

/* Quick health check pattern */
#define RBPF_WATCHDOG_CHECK_AND_RESET(ext, wd, obs)                         \
    do                                                                      \
    {                                                                       \
        RBPF_ResetReason _reason = rbpf_watchdog_check((ext), (wd), (obs)); \
        if (_reason != RESET_NONE)                                          \
        {                                                                   \
            rbpf_watchdog_reset((ext), (wd), _reason, NULL);                \
        }                                                                   \
    } while (0)

#endif /* RBPF_WATCHDOG_H */