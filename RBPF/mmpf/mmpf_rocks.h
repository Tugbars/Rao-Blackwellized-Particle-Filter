/**
 * @file mmpf_rocks.h
 * @brief IMM-MMPF-ROCKS: Interacting Multiple Model Particle Filter
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * ARCHITECTURE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Three parallel ROCKS (RBPF-KSC + Storvik) instances with different
 * structural hypotheses about volatility dynamics:
 *
 *   Calm:   φ=0.98, σ=0.10  (tight mean-reversion)
 *   Trend:  φ=0.95, σ=0.20  (momentum, moderate noise)
 *   Crisis: φ=0.80, σ=0.50  (explosive, low persistence)
 *
 * IMM (Interacting Multiple Model) mixing prevents hypothesis death:
 * - Before each step, particles are "teleported" between models
 * - Mixing weights μ[target][source] determined by transition matrix
 * - Ensures Crisis filter stays warm during calm markets
 *
 * OCSN-driven adaptive stickiness:
 * - High outlier fraction → lower stickiness → faster regime switching
 * - Low outlier fraction → higher stickiness → stable regimes
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * STEP SEQUENCE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   1. Update stickiness from t-1 OCSN outlier fraction
 *   2. Compute IMM mixing weights μ[target][source]
 *   3. Export particles from all 3 models to buffers
 *   4. Stratified resample: draw from combined pool into each model
 *   5. Import mixed particles back into models
 *   6. Step each RBPF (predict → update), get marginal likelihood
 *   7. Bayesian model weight update: w[k] *= L[k]
 *   8. Compute weighted volatility output
 *   9. Get OCSN outlier fraction for next tick
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * REFERENCES
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * - Blom & Bar-Shalom (1988): IMM algorithm
 * - Kim, Shephard & Chib (1998): KSC mixture for SV
 * - Storvik (2002): Online parameter learning via sufficient statistics
 *
 */

#ifndef MMPF_ROCKS_H
#define MMPF_ROCKS_H

#include "rbpf_ksc.h"
#include "rbpf_param_learn.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /*═══════════════════════════════════════════════════════════════════════════
     * CONSTANTS
     *═══════════════════════════════════════════════════════════════════════════*/

#define MMPF_N_MODELS 3
#define MMPF_ALIGN 64

    /*═══════════════════════════════════════════════════════════════════════════
     * HYPOTHESIS ENUMERATION
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef enum
    {
        MMPF_CALM = 0,  /* Tight mean-reversion, low vol-of-vol */
        MMPF_TREND = 1, /* Moderate persistence, medium vol-of-vol */
        MMPF_CRISIS = 2 /* Low persistence, high vol-of-vol */
    } MMPF_Hypothesis;

    /*═══════════════════════════════════════════════════════════════════════════
     * HYPOTHESIS PARAMETERS (Structural, NOT learned)
     *
     * These define the dynamics for each hypothesis:
     *   ℓ_t = μ + φ(ℓ_{t-1} - μ) + σ·η_t
     *
     * Only μ (mean level) is learned via Storvik. φ and σ are fixed per hypothesis.
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        rbpf_real_t mu_vol;    /* Long-run mean log-volatility */
        rbpf_real_t phi;       /* Persistence (mean-reversion speed = 1-φ) */
        rbpf_real_t sigma_eta; /* Vol-of-vol (innovation std dev) */
    } MMPF_HypothesisParams;

    /*═══════════════════════════════════════════════════════════════════════════
     * PARTICLE BUFFER (for IMM mixing workspace)
     *
     * Flat SoA storage for one model's worth of particles.
     * Allocated once in mmpf_create(), reused every tick.
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        int n_particles;
        int n_storvik_regimes; /* Storvik regime count (usually 4) */

        /* RBPF state (SoA) */
        rbpf_real_t *mu;         /* [n_particles] Kalman mean (log-vol) */
        rbpf_real_t *var;        /* [n_particles] Kalman variance */
        int *ksc_regime;         /* [n_particles] KSC mixture component (0-9) */
        rbpf_real_t *log_weight; /* [n_particles] Unnormalized log-weight */

        /* Storvik stats (SoA, particle-major: [particle * n_regimes + regime]) */
        param_real *storvik_m;     /* [n_particles * n_storvik_regimes] NIG mean */
        param_real *storvik_kappa; /* [n_particles * n_storvik_regimes] NIG precision count */
        param_real *storvik_alpha; /* [n_particles * n_storvik_regimes] NIG shape */
        param_real *storvik_beta;  /* [n_particles * n_storvik_regimes] NIG scale */
        param_real *storvik_mu;    /* [n_particles * n_storvik_regimes] Cached μ sample */
        param_real *storvik_sigma; /* [n_particles * n_storvik_regimes] Cached σ sample */

        /* Memory block (single allocation) */
        void *_memory;
        size_t _memory_size;

    } MMPF_ParticleBuffer;

    /*═══════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        /* Particle filter settings */
        int n_particles;       /* Particles per model (default: 512) */
        int n_ksc_regimes;     /* KSC mixture components (default: 4) */
        int n_storvik_regimes; /* Storvik parameter regimes (default: 4) */

        /* Hypothesis parameters */
        MMPF_HypothesisParams hypotheses[MMPF_N_MODELS];

        /* IMM transition matrix base settings */
        rbpf_real_t base_stickiness;   /* Default: 0.98 (98% stay in same regime) */
        rbpf_real_t min_stickiness;    /* Under stress: 0.85 */
        rbpf_real_t crisis_exit_boost; /* Crisis exits faster: 0.92 multiplier */
        rbpf_real_t min_mixing_prob;   /* Minimum transition probability (prevents lock-in): 0.01 */

        /* OCSN-driven adaptive stickiness */
        int enable_adaptive_stickiness; /* 1 = adjust stickiness based on outliers */

        /* Zero return handling (critical for HFT) */
        int zero_return_policy;        /* 0=skip update, 1=use floor, 2=censored interval */
        rbpf_real_t min_log_return_sq; /* Floor for log(r²) when r≈0 (default: -18.0) */

        /* Initial model weights */
        rbpf_real_t initial_weights[MMPF_N_MODELS];

        /* Robust OCSN settings (passed to each RBPF) */
        RBPF_RobustOCSN robust_ocsn;

        /* Storvik config (passed to each learner) */
        ParamLearnConfig storvik_config;

        /* RNG seed */
        uint64_t rng_seed;

    } MMPF_Config;

    /*═══════════════════════════════════════════════════════════════════════════
     * OUTPUT STRUCTURE
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        /* Weighted volatility estimates */
        rbpf_real_t volatility;     /* E[σ] = Σ w[k] × vol[k] */
        rbpf_real_t log_volatility; /* E[log(σ)] = Σ w[k] × log_vol[k] */
        rbpf_real_t volatility_std; /* Std[σ] across models - TOTAL (between + within) */

        /* Law of Total Variance components (for Kelly sizing diagnostics) */
        rbpf_real_t between_model_var; /* Var[E[V|M]] - disagreement between models */
        rbpf_real_t within_model_var;  /* E[Var[V|M]] - average uncertainty within each model */

        /* Model weights (sum to 1) */
        rbpf_real_t weights[MMPF_N_MODELS];

        /* Dominant model */
        MMPF_Hypothesis dominant;  /* argmax(weights) */
        rbpf_real_t dominant_prob; /* max(weights) */

        /* Per-model volatility estimates */
        rbpf_real_t model_vol[MMPF_N_MODELS];
        rbpf_real_t model_log_vol[MMPF_N_MODELS];
        rbpf_real_t model_log_vol_var[MMPF_N_MODELS]; /* Per-model Var[log(σ)] */
        rbpf_real_t model_likelihood[MMPF_N_MODELS];

        /* OCSN feedback */
        rbpf_real_t outlier_fraction;   /* From dominant model */
        rbpf_real_t current_stickiness; /* Adaptive stickiness value */

        /* IMM diagnostics */
        rbpf_real_t mixing_weights[MMPF_N_MODELS][MMPF_N_MODELS]; /* μ[target][source] */

        /* Per-model ESS */
        rbpf_real_t model_ess[MMPF_N_MODELS];

        /* Regime interpretation */
        int regime_stable;   /* 1 if dominant unchanged for N ticks */
        int ticks_in_regime; /* Consecutive ticks in current dominant */

        /* Zero return handling */
        int update_skipped; /* 1 if observation was treated as censored */

    } MMPF_Output;

    /*═══════════════════════════════════════════════════════════════════════════
     * MAIN STRUCTURE
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        /* Configuration */
        MMPF_Config config;
        int n_particles;

        /* Three independent RBPF + Storvik instances */
        RBPF_KSC *rbpf[MMPF_N_MODELS];
        ParamLearner *learner[MMPF_N_MODELS];

        /* Model weights (sum to 1) */
        rbpf_real_t weights[MMPF_N_MODELS];
        rbpf_real_t log_weights[MMPF_N_MODELS]; /* For numerical stability */

        /* IMM transition matrix (updated adaptively) */
        rbpf_real_t transition[MMPF_N_MODELS][MMPF_N_MODELS];

        /* Mixing weights μ[target][source] */
        rbpf_real_t mixing_weights[MMPF_N_MODELS][MMPF_N_MODELS];

        /* Particle buffers for IMM mixing (one per model) */
        MMPF_ParticleBuffer *buffer[MMPF_N_MODELS];

        /* Mixed particle buffer (target during stratified resample) */
        MMPF_ParticleBuffer *mixed_buffer[MMPF_N_MODELS];

        /* Stratified mixing counts */
        int mix_counts[MMPF_N_MODELS][MMPF_N_MODELS]; /* [target][source] */

        /* OCSN feedback state */
        rbpf_real_t outlier_fraction;
        rbpf_real_t current_stickiness;

        /* Cached outputs */
        rbpf_real_t weighted_vol;
        rbpf_real_t weighted_log_vol;
        rbpf_real_t weighted_vol_std;  /* Std across models (total: between + within) */
        rbpf_real_t between_model_var; /* Var[E[V|M]] - model disagreement */
        rbpf_real_t within_model_var;  /* E[Var[V|M]] - average model uncertainty */
        MMPF_Hypothesis dominant;

        /* Cached per-model outputs (populated during mmpf_step) */
        RBPF_KSC_Output model_output[MMPF_N_MODELS];
        rbpf_real_t model_likelihood[MMPF_N_MODELS];

        /* Regime stability tracking */
        MMPF_Hypothesis prev_dominant;
        int ticks_in_regime;

        /* Diagnostics */
        uint64_t total_steps;
        uint64_t regime_switches;
        uint64_t imm_mix_count;

        /* RNG for stratified resampling */
        rbpf_pcg32_t rng;

        /* Robust OCSN (shared reference for all models) */
        RBPF_RobustOCSN robust_ocsn;

    } MMPF_ROCKS;

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Configuration
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Get default configuration.
     *
     * Defaults:
     *   - 512 particles per model
     *   - Calm: φ=0.98, σ=0.10
     *   - Trend: φ=0.95, σ=0.20
     *   - Crisis: φ=0.80, σ=0.50
     *   - Base stickiness: 0.98
     *   - Adaptive stickiness: enabled
     */
    MMPF_Config mmpf_config_defaults(void);

    /**
     * Get HFT-optimized configuration.
     *
     * Lower particle count, aggressive settings.
     */
    MMPF_Config mmpf_config_hft(void);

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Lifecycle
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Create MMPF instance.
     *
     * Allocates 3 RBPF instances, 3 ParamLearners, 6 particle buffers.
     *
     * @param config  Configuration (NULL for defaults)
     * @return        MMPF instance, or NULL on failure
     */
    MMPF_ROCKS *mmpf_create(const MMPF_Config *config);

    /**
     * Destroy MMPF instance.
     */
    void mmpf_destroy(MMPF_ROCKS *mmpf);

    /**
     * Reset to initial state.
     *
     * @param mmpf         MMPF instance
     * @param initial_vol  Initial volatility estimate (e.g., 0.02 for 2%)
     */
    void mmpf_reset(MMPF_ROCKS *mmpf, rbpf_real_t initial_vol);

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Main Step
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Process one observation.
     *
     * Full sequence:
     *   1. Update stickiness from t-1 OCSN
     *   2. Compute mixing weights
     *   3. Export particles from all models
     *   4. Stratified resample into each model
     *   5. Import mixed particles
     *   6. Step each RBPF (returns marginal likelihood)
     *   7. Bayesian weight update
     *   8. Compute weighted output
     *   9. Get OCSN for next tick
     *
     * @param mmpf    MMPF instance
     * @param y       Observation: log(r²) where r is return
     * @param output  Output structure (can be NULL if only volatility needed)
     */
    void mmpf_step(MMPF_ROCKS *mmpf, rbpf_real_t y, MMPF_Output *output);

    /**
     * Step with APF (auxiliary particle filter) lookahead.
     *
     * Uses next observation for improved regime change detection.
     *
     * @param mmpf      MMPF instance
     * @param y_current Current observation
     * @param y_next    Next observation (lookahead)
     * @param output    Output structure
     */
    void mmpf_step_apf(MMPF_ROCKS *mmpf, rbpf_real_t y_current, rbpf_real_t y_next,
                       MMPF_Output *output);

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Outputs
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Get weighted volatility estimate.
     * σ = Σ w[k] × σ[k]
     */
    rbpf_real_t mmpf_get_volatility(const MMPF_ROCKS *mmpf);

    /**
     * Get weighted log-volatility estimate.
     * log(σ) = Σ w[k] × log(σ[k])
     */
    rbpf_real_t mmpf_get_log_volatility(const MMPF_ROCKS *mmpf);

    /**
     * Get volatility uncertainty (std across models).
     */
    rbpf_real_t mmpf_get_volatility_std(const MMPF_ROCKS *mmpf);

    /**
     * Get dominant hypothesis.
     */
    MMPF_Hypothesis mmpf_get_dominant(const MMPF_ROCKS *mmpf);

    /**
     * Get probability of dominant hypothesis.
     */
    rbpf_real_t mmpf_get_dominant_probability(const MMPF_ROCKS *mmpf);

    /**
     * Get model weights.
     *
     * @param mmpf     MMPF instance
     * @param weights  Output array [MMPF_N_MODELS]
     */
    void mmpf_get_weights(const MMPF_ROCKS *mmpf, rbpf_real_t *weights);

    /**
     * Get OCSN outlier fraction (from dominant model).
     */
    rbpf_real_t mmpf_get_outlier_fraction(const MMPF_ROCKS *mmpf);

    /**
     * Get current adaptive stickiness value.
     */
    rbpf_real_t mmpf_get_stickiness(const MMPF_ROCKS *mmpf);

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Model Access
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Get volatility from specific model.
     */
    rbpf_real_t mmpf_get_model_volatility(const MMPF_ROCKS *mmpf, MMPF_Hypothesis model);

    /**
     * Get ESS from specific model.
     */
    rbpf_real_t mmpf_get_model_ess(const MMPF_ROCKS *mmpf, MMPF_Hypothesis model);

    /**
     * Get underlying RBPF instance (for advanced access).
     */
    const RBPF_KSC *mmpf_get_rbpf(const MMPF_ROCKS *mmpf, MMPF_Hypothesis model);

    /**
     * Get underlying ParamLearner (for advanced access).
     */
    const ParamLearner *mmpf_get_learner(const MMPF_ROCKS *mmpf, MMPF_Hypothesis model);

    /*═══════════════════════════════════════════════════════════════════════════
     * API: IMM Control
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Set IMM transition matrix directly.
     *
     * @param mmpf        MMPF instance
     * @param transition  3x3 matrix [from][to], row-major
     */
    void mmpf_set_transition_matrix(MMPF_ROCKS *mmpf, const rbpf_real_t *transition);

    /**
     * Set base stickiness (diagonal of transition matrix).
     */
    void mmpf_set_stickiness(MMPF_ROCKS *mmpf, rbpf_real_t base, rbpf_real_t min);

    /**
     * Enable/disable adaptive stickiness.
     */
    void mmpf_set_adaptive_stickiness(MMPF_ROCKS *mmpf, int enable);

    /**
     * Force model weights (for testing/initialization).
     *
     * @param mmpf     MMPF instance
     * @param weights  New weights [MMPF_N_MODELS], will be normalized
     */
    void mmpf_set_weights(MMPF_ROCKS *mmpf, const rbpf_real_t *weights);

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Diagnostics
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Print summary to stdout.
     */
    void mmpf_print_summary(const MMPF_ROCKS *mmpf);

    /**
     * Print detailed output structure.
     */
    void mmpf_print_output(const MMPF_Output *output);

    /**
     * Get diagnostic counters.
     */
    void mmpf_get_diagnostics(const MMPF_ROCKS *mmpf,
                              uint64_t *total_steps,
                              uint64_t *regime_switches,
                              uint64_t *imm_mix_count);

    /*═══════════════════════════════════════════════════════════════════════════
     * PARTICLE BUFFER API (Internal, but exposed for testing)
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Create particle buffer.
     */
    MMPF_ParticleBuffer *mmpf_buffer_create(int n_particles, int n_storvik_regimes);

    /**
     * Destroy particle buffer.
     */
    void mmpf_buffer_destroy(MMPF_ParticleBuffer *buf);

    /**
     * Export particles from RBPF + Storvik to buffer.
     */
    void mmpf_buffer_export(MMPF_ParticleBuffer *buf,
                            const RBPF_KSC *rbpf,
                            const ParamLearner *learner);

    /**
     * Import particles from buffer to RBPF + Storvik.
     */
    void mmpf_buffer_import(const MMPF_ParticleBuffer *buf,
                            RBPF_KSC *rbpf,
                            ParamLearner *learner);

    /**
     * Copy single particle from src buffer to dst buffer.
     *
     * @param dst       Destination buffer
     * @param dst_idx   Destination particle index
     * @param src       Source buffer
     * @param src_idx   Source particle index
     */
    void mmpf_buffer_copy_particle(MMPF_ParticleBuffer *dst, int dst_idx,
                                   const MMPF_ParticleBuffer *src, int src_idx);

#ifdef __cplusplus
}
#endif

#endif /* MMPF_ROCKS_H */