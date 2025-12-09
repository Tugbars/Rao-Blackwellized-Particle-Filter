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

    /*─────────────────────────────────────────────────────────────────────────────
     * PARAMETER LEARNING MODE
     *───────────────────────────────────────────────────────────────────────────*/

    typedef enum
    {
        RBPF_PARAM_DISABLED = 0, /* No online learning (use fixed params) */
        RBPF_PARAM_LIU_WEST = 1, /* Liu-West only (original) */
        RBPF_PARAM_STORVIK = 2,  /* Sleeping Storvik only */
        RBPF_PARAM_HYBRID = 3    /* Both (Storvik for posterior, Liu-West for fast) */
    } RBPF_ParamMode;

    /*─────────────────────────────────────────────────────────────────────────────
     * EXTENDED RBPF STRUCTURE
     *
     * Wraps RBPF_KSC with additional Storvik parameter learning.
     * Use rbpf_ext_* functions instead of rbpf_ksc_* for integrated behavior.
     *───────────────────────────────────────────────────────────────────────────*/

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

        /*═══════════════════════════════════════════════════════════════════════
         * TRANSITION MATRIX LEARNING (Optional)
         *
         * Learns P_ij (probability of regime i → j) from particle transitions.
         * Uses Dirichlet-Multinomial sufficient statistics with forgetting.
         *
         * Benefits:
         *   - "Choppy" markets: learns looser diagonal → faster switching
         *   - "Crisis Persist": learns sticky R3 → doesn't jump back prematurely
         *═══════════════════════════════════════════════════════════════════════*/
        int trans_learn_enabled;                                 /* 0=fixed, 1=adaptive */
        double trans_counts[RBPF_MAX_REGIMES][RBPF_MAX_REGIMES]; /* Sufficient stats N_ij */
        double trans_forgetting;                                 /* Decay factor (0.995 default) */
        double trans_prior_diag;                                 /* Prior strength for diagonal (50.0) */
        double trans_prior_off;                                  /* Prior strength for off-diag (1.0) */
        int trans_update_interval;                               /* Rebuild LUT every N ticks (100) */
        int trans_ticks_since_update;                            /* Counter */

    } RBPF_Extended;

    /*─────────────────────────────────────────────────────────────────────────────
     * LIFECYCLE
     *───────────────────────────────────────────────────────────────────────────*/

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

    /*─────────────────────────────────────────────────────────────────────────────
     * CONFIGURATION
     *───────────────────────────────────────────────────────────────────────────*/

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

    /*─────────────────────────────────────────────────────────────────────────────
     * TRANSITION MATRIX LEARNING (Optional)
     *───────────────────────────────────────────────────────────────────────────*/

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

    /*─────────────────────────────────────────────────────────────────────────────
     * MAIN UPDATE - THE HOT PATH
     *───────────────────────────────────────────────────────────────────────────*/

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

    /*─────────────────────────────────────────────────────────────────────────────
     * PARAMETER ACCESS
     *───────────────────────────────────────────────────────────────────────────*/

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

    /*─────────────────────────────────────────────────────────────────────────────
     * DEBUG
     *───────────────────────────────────────────────────────────────────────────*/

    void rbpf_ext_print_config(const RBPF_Extended *ext);
    void rbpf_ext_print_storvik_stats(const RBPF_Extended *ext, int regime);

#ifdef __cplusplus
}
#endif

#endif /* RBPF_KSC_PARAM_INTEGRATION_H */