/**
 * @file rbpf_ksc_param_integration.h
 * @brief Integration layer: RBPF-KSC + Sleeping Storvik Parameter Learning
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * DESIGN CHOICE: STORVIK VS LIU-WEST
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Liu-West (current):
 *   + Fast adaptation (shrinkage toward mean)
 *   + Simple implementation
 *   - Point estimates only
 *   - Only learns during resample
 *   - Regime starvation (minority regimes don't learn)
 *
 * Sleeping Storvik (new):
 *   + Full Bayesian posterior (NIG conjugate)
 *   + Always updates stats (learns every tick)
 *   + Regime-aware (each regime learns independently)
 *   + Adaptive sampling (sleep in calm, wake in crisis)
 *   - More complex
 *   - Slightly higher memory (sufficient statistics per particle)
 *
 * This integration provides THREE modes:
 *
 *   RBPF_PARAM_LIU_WEST:   Original Liu-West (default, backward compatible)
 *   RBPF_PARAM_STORVIK:    Sleeping Storvik only
 *   RBPF_PARAM_HYBRID:     Both running (Storvik tracks, Liu-West adapts)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#ifndef RBPF_KSC_PARAM_INTEGRATION_H
#define RBPF_KSC_PARAM_INTEGRATION_H

#include "rbpf_ksc.h"
#include "rbpf_param_learn.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /*═══════════════════════════════════════════════════════════════════════════
     * ENUMS (must come first - used by structs below)
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Parameter learning mode
     */
    typedef enum
    {
        RBPF_PARAM_DISABLED = 0, /* No online learning (use fixed params) */
        RBPF_PARAM_LIU_WEST = 1, /* Liu-West only (original) */
        RBPF_PARAM_STORVIK = 2,  /* Sleeping Storvik only */
        RBPF_PARAM_HYBRID = 3    /* Both (Storvik for posterior, Liu-West for fast) */
    } RBPF_ParamMode;

    /**
     * Asset class presets
     *
     * Pre-tuned parameter sets for different asset classes.
     * Reduces tuning burden while providing sensible defaults.
     */
    typedef enum
    {
        RBPF_PRESET_EQUITY_INDEX = 0, /* SPY, QQQ, ES: moderate fat tails */
        RBPF_PRESET_SINGLE_STOCK,     /* AAPL, TSLA: higher vol, more outliers */
        RBPF_PRESET_FX_G10,           /* EUR/USD, USD/JPY: thinner tails */
        RBPF_PRESET_FX_EM,            /* USD/MXN, USD/TRY: fatter tails */
        RBPF_PRESET_CRYPTO,           /* BTC, ETH: extreme tails */
        RBPF_PRESET_COMMODITIES,      /* CL, GC: moderate tails, jump risk */
        RBPF_PRESET_BONDS,            /* ZN, ZB: thin tails, rare jumps */
        RBPF_PRESET_CUSTOM            /* User-defined */
    } RBPF_AssetPreset;

    /*═══════════════════════════════════════════════════════════════════════════
     * ROBUST OCSN (11TH COMPONENT)
     *
     * NOTE: RBPF_OutlierParams and RBPF_RobustOCSN are defined in rbpf_ksc.h
     *═══════════════════════════════════════════════════════════════════════════*/

    /*═══════════════════════════════════════════════════════════════════════════
     * HAWKES SELF-EXCITING PROCESS
     *
     * Models volatility clustering: "shocks breed shocks"
     * λ(t) = μ + (λ(t-1) - μ) × e^(-β) + α × I(|r| > threshold)
     *
     * High intensity → boost upward regime transitions
     *
     * ADAPTIVE DECAY: β varies by current regime to prevent:
     *   - Sticky crisis (R3 with slow decay → phantom regime)
     *   - Premature calm (R0 with fast decay → misses clustering)
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        int enabled;

        /* Core parameters */
        rbpf_real_t mu;        /* Baseline intensity */
        rbpf_real_t alpha;     /* Jump size on excitation */
        rbpf_real_t beta;      /* Base decay rate (half-life ≈ 0.693/β) */
        rbpf_real_t threshold; /* |return| threshold to trigger jump */

        /* ADAPTIVE DECAY: Per-regime β multipliers
         * Effective β = beta * beta_regime_scale[regime]
         * Higher scale = faster decay = shorter memory */
        rbpf_real_t beta_regime_scale[RBPF_MAX_REGIMES];
        int adaptive_beta_enabled;

        /* State */
        rbpf_real_t intensity;      /* Current λ(t) */
        rbpf_real_t intensity_prev; /* Previous λ(t-1) for hysteresis */

        /* Transition modification */
        rbpf_real_t boost_scale; /* How much intensity affects transitions (default: 0.1) */
        rbpf_real_t boost_cap;   /* Maximum boost (default: 0.25) */

        /* Efficiency: track if LUT needs rebuild */
        int lut_dirty; /* 1 if we modified LUT and need to restore */
    } RBPF_HawkesState;

    /*═══════════════════════════════════════════════════════════════════════════
     * EXTENDED RBPF STRUCTURE
     *
     * Wraps RBPF_KSC with additional Storvik parameter learning.
     * Use rbpf_ext_* functions instead of rbpf_ksc_* for integrated behavior.
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        /* Core RBPF-KSC filter */
        RBPF_KSC *rbpf;

        /* Sleeping Storvik parameter learner */
        ParamLearner storvik;
        int storvik_initialized;

        /* Mode selection */
        RBPF_ParamMode param_mode;

        /* Workspace for particle info extraction */
        ParticleInfo *particle_info; /* [n_particles] - reused each tick */
        int *prev_regime;            /* [n_particles] - track regime changes */

        /* Lag buffer for ell_lag (log-vol from previous tick) */
        rbpf_real_t *ell_lag_buffer; /* [n_particles] */

        /* SSA bridge connection (optional) */
        int ssa_connected;
        int structural_break_signaled;

        /*───────────────────────────────────────────────────────────────────────
         * TRANSITION MATRIX LEARNING (Optional)
         *
         * Learns P_ij (probability of regime i → j) from particle transitions.
         * Uses Dirichlet-Multinomial sufficient statistics with forgetting.
         *
         * Benefits:
         *   - "Choppy" markets: learns looser diagonal → faster switching
         *   - "Crisis Persist": learns sticky R3 → doesn't jump back prematurely
         *─────────────────────────────────────────────────────────────────────*/
        int trans_learn_enabled;                                 /* 0=fixed, 1=adaptive */
        double trans_counts[RBPF_MAX_REGIMES][RBPF_MAX_REGIMES]; /* Sufficient stats N_ij */
        double trans_forgetting;                                 /* Decay factor (0.995 default) */
        double trans_prior_diag;                                 /* Prior strength for diagonal (50.0) */
        double trans_prior_off;                                  /* Prior strength for off-diag (1.0) */
        int trans_update_interval;                               /* Rebuild LUT every N ticks (100) */
        int trans_ticks_since_update;                            /* Counter */

        /*───────────────────────────────────────────────────────────────────────
         * HAWKES SELF-EXCITATION (Optional)
         *─────────────────────────────────────────────────────────────────────*/
        RBPF_HawkesState hawkes;
        rbpf_real_t base_trans_matrix[RBPF_MAX_REGIMES * RBPF_MAX_REGIMES];

        /*───────────────────────────────────────────────────────────────────────
         * ROBUST OCSN - 11TH COMPONENT (Optional)
         *─────────────────────────────────────────────────────────────────────*/
        RBPF_RobustOCSN robust_ocsn;

        /*───────────────────────────────────────────────────────────────────────
         * ASSET PRESET & DIAGNOSTICS
         *─────────────────────────────────────────────────────────────────────*/
        RBPF_AssetPreset current_preset;
        uint64_t tick_count;
        rbpf_real_t last_hawkes_intensity;
        rbpf_real_t last_outlier_fraction;

    } RBPF_Extended;

    /*═══════════════════════════════════════════════════════════════════════════
     * LIFECYCLE
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Create extended RBPF with parameter learning
     *
     * @param n_particles  Number of particles
     * @param n_regimes    Number of regimes (max 8)
     * @param mode         Parameter learning mode
     * @return Extended RBPF handle, or NULL on failure
     */
    RBPF_Extended *rbpf_ext_create(int n_particles, int n_regimes, RBPF_ParamMode mode);

    /**
     * Destroy extended RBPF
     */
    void rbpf_ext_destroy(RBPF_Extended *ext);

    /**
     * Initialize filter state
     *
     * @param ext   Extended RBPF handle
     * @param mu0   Initial log-volatility mean
     * @param var0  Initial log-volatility variance
     */
    void rbpf_ext_init(RBPF_Extended *ext, rbpf_real_t mu0, rbpf_real_t var0);

    /*═══════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Set regime parameters (forwards to both RBPF and Storvik)
     */
    void rbpf_ext_set_regime_params(RBPF_Extended *ext, int regime,
                                    rbpf_real_t theta, rbpf_real_t mu_vol, rbpf_real_t sigma_vol);

    /**
     * Build transition LUT (forwards to RBPF)
     */
    void rbpf_ext_build_transition_lut(RBPF_Extended *ext, const rbpf_real_t *trans_matrix);

    /**
     * Configure Storvik sampling intervals per regime
     *
     * @param ext       Extended RBPF handle
     * @param regime    Regime index
     * @param interval  Ticks between samples (e.g., 50 for R0, 1 for R3)
     */
    void rbpf_ext_set_storvik_interval(RBPF_Extended *ext, int regime, int interval);

    /**
     * Configure Storvik for HFT mode (less frequent sampling in calm markets)
     */
    void rbpf_ext_set_hft_mode(RBPF_Extended *ext, int enable);

    /**
     * Signal structural break from SSA/BOCPD (triggers immediate sampling)
     */
    void rbpf_ext_signal_structural_break(RBPF_Extended *ext);

    /*═══════════════════════════════════════════════════════════════════════════
     * TRANSITION MATRIX LEARNING
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Enable/disable online transition matrix learning
     *
     * When enabled, the filter learns P_ij from observed particle transitions.
     * This helps in:
     *   - Choppy markets: learns to switch regimes more eagerly
     *   - Crisis persistence: learns that crisis is sticky
     *
     * @param ext     Extended RBPF handle
     * @param enable  1 to enable, 0 to disable
     */
    void rbpf_ext_enable_transition_learning(RBPF_Extended *ext, int enable);

    /**
     * Configure transition learning parameters
     *
     * @param ext              Extended RBPF handle
     * @param forgetting       Decay factor per tick (0.995 = slow adaptation)
     * @param prior_diag       Prior strength for diagonal (50.0 = sticky)
     * @param prior_off        Prior strength for off-diagonal (1.0)
     * @param update_interval  Ticks between LUT rebuilds (100)
     */
    void rbpf_ext_configure_transition_learning(RBPF_Extended *ext,
                                                double forgetting,
                                                double prior_diag,
                                                double prior_off,
                                                int update_interval);

    /**
     * Reset transition counts to zero (keeps priors)
     */
    void rbpf_ext_reset_transition_counts(RBPF_Extended *ext);

    /**
     * Get current learned transition probability
     *
     * @param ext   Extended RBPF handle
     * @param from  Source regime
     * @param to    Destination regime
     * @return Learned P(from → to)
     */
    double rbpf_ext_get_transition_prob(const RBPF_Extended *ext, int from, int to);

    /*═══════════════════════════════════════════════════════════════════════════
     * HAWKES SELF-EXCITATION
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Enable Hawkes self-exciting process for transition modulation
     *
     * When enabled, large returns increase intensity, which boosts
     * the probability of transitioning to higher volatility regimes.
     *
     * @param ext       Extended RBPF handle
     * @param mu        Baseline intensity (e.g., 0.05)
     * @param alpha     Jump size on excitation (e.g., 0.3)
     * @param beta      Decay rate (e.g., 0.1 → half-life ~7 ticks)
     * @param threshold Return magnitude to trigger excitation (e.g., 0.03 = 3%)
     */
    void rbpf_ext_enable_hawkes(RBPF_Extended *ext,
                                rbpf_real_t mu, rbpf_real_t alpha,
                                rbpf_real_t beta, rbpf_real_t threshold);

    /**
     * Disable Hawkes process
     */
    void rbpf_ext_disable_hawkes(RBPF_Extended *ext);

    /**
     * Set Hawkes transition boost parameters
     *
     * @param ext         Extended RBPF handle
     * @param boost_scale How much intensity affects transitions (default: 0.1)
     * @param boost_cap   Maximum probability boost (default: 0.25)
     */
    void rbpf_ext_set_hawkes_boost(RBPF_Extended *ext,
                                   rbpf_real_t boost_scale, rbpf_real_t boost_cap);

    /**
     * Enable adaptive Hawkes decay (regime-dependent β)
     *
     * When enabled, decay rate varies by current regime:
     *   R0 (Calm):    Fast decay (short memory) - quickly forget flash crashes
     *   R3 (Crisis):  Slow decay (long memory) - crisis persists
     *
     * This prevents "phantom regime" (stuck in R3 after flash crash)
     * while preserving volatility clustering during true crises.
     *
     * @param ext     Extended RBPF handle
     * @param enable  1 to enable, 0 to disable
     */
    void rbpf_ext_enable_adaptive_hawkes(RBPF_Extended *ext, int enable);

    /**
     * Set per-regime Hawkes decay multiplier
     *
     * Effective β = base_β × scale[regime]
     * Higher scale = faster decay = shorter memory
     *
     * @param ext     Extended RBPF handle
     * @param regime  Regime index
     * @param scale   Multiplier (default: R0=2.0, R1=1.5, R2=1.0, R3=0.5)
     */
    void rbpf_ext_set_hawkes_regime_scale(RBPF_Extended *ext, int regime, rbpf_real_t scale);

    /**
     * Get current Hawkes intensity
     */
    rbpf_real_t rbpf_ext_get_hawkes_intensity(const RBPF_Extended *ext);

    /*═══════════════════════════════════════════════════════════════════════════
     * ROBUST OCSN (11TH OUTLIER COMPONENT)
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Enable robust OCSN (11th outlier component)
     *
     * Adds a wide Gaussian "safety valve" to handle tail events.
     * Uses default per-regime parameters.
     *
     * @param ext  Extended RBPF handle
     */
    void rbpf_ext_enable_robust_ocsn(RBPF_Extended *ext);

    /**
     * Enable robust OCSN with custom parameters (same for all regimes)
     *
     * @param ext       Extended RBPF handle
     * @param prob      Outlier probability (e.g., 0.01 = 1%)
     * @param variance  Outlier variance (e.g., 20.0)
     */
    void rbpf_ext_enable_robust_ocsn_simple(RBPF_Extended *ext,
                                            rbpf_real_t prob, rbpf_real_t variance);

    /**
     * Set per-regime outlier parameters
     *
     * @param ext       Extended RBPF handle
     * @param regime    Regime index
     * @param prob      Outlier probability for this regime
     * @param variance  Outlier variance for this regime
     */
    void rbpf_ext_set_outlier_params(RBPF_Extended *ext, int regime,
                                     rbpf_real_t prob, rbpf_real_t variance);

    /**
     * Disable robust OCSN
     */
    void rbpf_ext_disable_robust_ocsn(RBPF_Extended *ext);

    /**
     * Get outlier fraction from last observation
     *
     * @return Fraction of likelihood explained by outlier component (0-1)
     */
    rbpf_real_t rbpf_ext_get_outlier_fraction(const RBPF_Extended *ext);

    /*═══════════════════════════════════════════════════════════════════════════
     * ASSET PRESETS
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Apply asset class preset
     *
     * Configures Hawkes and Robust OCSN parameters for specific asset class.
     * Call this AFTER rbpf_ext_create() and BEFORE rbpf_ext_init().
     *
     * @param ext     Extended RBPF handle
     * @param preset  Asset class preset
     */
    void rbpf_ext_apply_preset(RBPF_Extended *ext, RBPF_AssetPreset preset);

    /**
     * Get current preset (for diagnostics)
     */
    RBPF_AssetPreset rbpf_ext_get_preset(const RBPF_Extended *ext);

    /*═══════════════════════════════════════════════════════════════════════════
     * MAIN UPDATE - THE HOT PATH
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Process one observation (integrated parameter learning)
     *
     * This is the main entry point. It:
     * 1. Runs RBPF-KSC Kalman update
     * 2. Extracts particle info for Storvik
     * 3. Updates Storvik sufficient statistics
     * 4. Conditionally samples new parameters
     * 5. Syncs learned params back to RBPF (if Storvik mode)
     *
     * @param ext     Extended RBPF handle
     * @param obs     Observation (return, NOT log-return)
     * @param output  Output structure (filled on return)
     */
    void rbpf_ext_step(RBPF_Extended *ext, rbpf_real_t obs, RBPF_KSC_Output *output);

    /**
     * Process with APF lookahead (for regime change detection)
     */
    void rbpf_ext_step_apf(RBPF_Extended *ext, rbpf_real_t obs_current,
                           rbpf_real_t obs_next, RBPF_KSC_Output *output);

    /*═══════════════════════════════════════════════════════════════════════════
     * PARAMETER ACCESS
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Get learned parameters (from Storvik or Liu-West based on mode)
     *
     * @param ext     Extended RBPF handle
     * @param regime  Regime index
     * @param mu_vol  Output: learned μ_vol
     * @param sigma_vol Output: learned σ_vol
     */
    void rbpf_ext_get_learned_params(const RBPF_Extended *ext, int regime,
                                     rbpf_real_t *mu_vol, rbpf_real_t *sigma_vol);

    /**
     * Get Storvik posterior summary (full Bayesian info)
     *
     * Only available in STORVIK or HYBRID mode.
     *
     * @param ext     Extended RBPF handle
     * @param regime  Regime index
     * @param summary Output: posterior summary
     */
    void rbpf_ext_get_storvik_summary(const RBPF_Extended *ext, int regime,
                                      RegimeParams *summary);

    /**
     * Get learning statistics
     */
    void rbpf_ext_get_learning_stats(const RBPF_Extended *ext,
                                     uint64_t *stat_updates,
                                     uint64_t *samples_drawn,
                                     uint64_t *samples_skipped);

    /*═══════════════════════════════════════════════════════════════════════════
     * DEBUG
     *═══════════════════════════════════════════════════════════════════════════*/

    void rbpf_ext_print_config(const RBPF_Extended *ext);
    void rbpf_ext_print_storvik_stats(const RBPF_Extended *ext, int regime);

#ifdef __cplusplus
}
#endif

#endif /* RBPF_KSC_PARAM_INTEGRATION_H */