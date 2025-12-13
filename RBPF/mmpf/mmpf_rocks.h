/**
 * @file mmpf_rocks.h
 * @brief IMM-MMPF-ROCKS: Interacting Multiple Model Particle Filter
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * ARCHITECTURE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Three parallel RBPF_Extended instances with different structural hypotheses
 * about volatility dynamics:
 *
 *   Calm:   φ=0.98, σ=0.10  (tight mean-reversion)
 *   Trend:  φ=0.95, σ=0.20  (momentum, moderate noise)
 *   Crisis: φ=0.80, σ=0.50  (explosive, low persistence)
 *
 * Each RBPF_Extended bundles:
 *   - RBPF_KSC: Rao-Blackwellized particle filter with KSC mixture
 *   - Storvik: Online parameter learning
 *   - OCSN: Outlier Component Selection Network (per-hypothesis!)
 *
 * WHY PER-HYPOTHESIS OCSN:
 * Same observation, different interpretations. A 5% move is 6σ under Calm
 * but only 1.4σ under Crisis. Each hypothesis must judge outliers under
 * its own world view.
 *
 * WHY 3 RBPFS INSTEAD OF 1:
 * A single RBPF adapts slowly (~100 ticks). With 3 parallel hypotheses,
 * one is always "warm" with appropriate dynamics. Crisis hits → just
 * upweight Crisis-RBPF → instant response.
 *
 * IMM (Interacting Multiple Model) mixing prevents hypothesis death:
 * - Before each step, particles are "teleported" between models
 * - Mixing weights μ[target][source] determined by transition matrix
 * - Ensures Crisis filter stays warm during calm markets
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
 *   6. Step each RBPF_Extended (predict → update), get marginal likelihood
 *   7. Bayesian model weight update: w[k] *= L[k]
 *   8. Compute weighted volatility output
 *   9. Get per-model OCSN outlier fraction for next tick
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

#ifdef MMPF_USE_TEST_STUB
/* Use simplified test stub for compiling without full MKL dependencies */
/* Note: rocks_test_stub.c must be compiled together */
#else
#include "rbpf_ksc_param_integration.h" /* RBPF_Extended: RBPF + Storvik + OCSN bundled */
#endif

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

        /* Storvik parameter learning sync
         *
         * When enabled (default=1): Storvik learned params → RBPF particle_mu_vol
         * When disabled (0): RBPF uses fixed hypothesis params, no learning adaptation
         *
         * Disable for unit tests that verify hypothesis discrimination - learning
         * causes all models to converge toward similar params, destroying discrimination.
         */
        int enable_storvik_sync;

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

        /*═══════════════════════════════════════════════════════════════════════
         * GLOBAL BASELINE ("Climate vs Weather")
         *
         * The key insight: Learn ONE global baseline that tracks secular drift,
         * then define hypotheses as FIXED OFFSETS from that baseline.
         *
         * μ_base = EWMA(weighted_log_vol)     ← adapts to decade
         * μ_calm   = μ_base + offset_calm     ← fixed structure
         * μ_trend  = μ_base + offset_trend    ← fixed structure
         * μ_crisis = μ_base + offset_crisis   ← fixed structure
         *
         * This prevents the Icarus Paradox: hypotheses can't converge because
         * their offsets are constants. Discrimination is structurally guaranteed.
         *═══════════════════════════════════════════════════════════════════════*/

        int enable_global_baseline;                /* 1 = use adaptive baseline (recommended) */
        rbpf_real_t global_mu_vol_init;            /* Initial baseline (e.g., log(0.01) = -4.6) */
        rbpf_real_t global_mu_vol_alpha;           /* EWMA decay (0.999 = ~1000 tick half-life) */
        rbpf_real_t mu_vol_offsets[MMPF_N_MODELS]; /* Relative offsets from baseline */

        /*═══════════════════════════════════════════════════════════════════════
         * FAIR WEATHER GATE
         *
         * Prevents baseline corruption during crisis events. Crisis spikes are
         * "weather" (temporary), not "climate" (structural). If we let the baseline
         * drift up during COVID crash, it takes months to drift back down.
         *
         * Solution: Freeze baseline updates when w_crisis > threshold.
         * Uses hysteresis to prevent flickering near the threshold.
         *
         * Example with defaults (gate_on=0.5, gate_off=0.4):
         *   - Normal: w_crisis=0.1 → baseline updates normally
         *   - Crisis hits: w_crisis=0.6 → baseline FROZEN
         *   - Crisis persists: w_crisis=0.45 → still frozen (hysteresis)
         *   - Crisis ends: w_crisis=0.35 → baseline UNFROZEN, resumes updating
         *═══════════════════════════════════════════════════════════════════════*/

        rbpf_real_t baseline_gate_on;  /* Freeze when w_crisis > this (default: 0.5) */
        rbpf_real_t baseline_gate_off; /* Unfreeze when w_crisis < this (default: 0.4) */

        /*═══════════════════════════════════════════════════════════════════════
         * GATED DYNAMICS LEARNING
         *
         * When enabled, each hypothesis learns its OWN φ and σ_η from data
         * where IT is dominant. Prevents "pollution" of Crisis learning from
         * Calm data.
         *═══════════════════════════════════════════════════════════════════════*/

        int enable_gated_learning;            /* 1 = weight dynamics updates by regime prob */
        rbpf_real_t gated_learning_threshold; /* Min weight to update (0.0 = soft gate) */

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

        /* Global baseline diagnostics */
        rbpf_real_t global_mu_vol; /* Current baseline (for monitoring secular drift) */
        int baseline_frozen;       /* 1 if baseline is frozen (Fair Weather Gate active) */

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

        /*───────────────────────────────────────────────────────────────────────
         * THREE INDEPENDENT RBPF_EXTENDED INSTANCES
         *
         * Each RBPF_Extended bundles: RBPF_KSC + Storvik + OCSN
         *
         * Why per-hypothesis OCSN matters:
         *   Same observation, different interpretations:
         *
         *   | Hypothesis | μ_vol anchor | 5% move | OCSN verdict |
         *   |------------|--------------|---------|--------------|
         *   | Calm       | -4.83 (0.8%) | 6σ      | Outlier!     |
         *   | Crisis     | -3.35 (3.5%) | 1.4σ    | Normal       |
         *
         *   With shared OCSN, Crisis-RBPF would get confused by Calm-RBPF's
         *   outlier signal. Each hypothesis must judge outliers under its
         *   own world view.
         *─────────────────────────────────────────────────────────────────────*/
        RBPF_Extended *ext[MMPF_N_MODELS];

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

        /* Note: OCSN is now per-hypothesis inside each RBPF_Extended.
         * Access via: mmpf->ext[k]->robust_ocsn
         * The outlier_fraction below is the weighted average across hypotheses. */

        /*═══════════════════════════════════════════════════════════════════════════
         * BOCPD SHOCK MECHANISM
         *═══════════════════════════════════════════════════════════════════════════
         *
         * When BOCPD detects a changepoint (posterior collapse), MMPF's normal sticky
         * transitions (98% stay in same regime) slow down adaptation. The shock
         * mechanism temporarily forces exploration:
         *
         * PROBLEM:
         *   - Normal transitions: Calm→Calm = 98%, Calm→Crisis = 1%
         *   - After real regime change, takes ~100 ticks for weights to shift
         *   - Too slow for trading response
         *
         * SOLUTION (Sidecar Architecture):
         *   1. BOCPD runs in parallel, detects "something changed"
         *   2. On changepoint signal, inject shock into MMPF for ONE tick:
         *      - Override transitions to uniform (33% each direction)
         *      - Boost process noise 50x (particles explore wider μ_vol range)
         *   3. Run mmpf_step() - likelihoods now determine winner immediately
         *   4. Restore normal transitions
         *
         * RESULT:
         *   - Detection lag: ~100 ticks → <20 ticks
         *   - False positive protection: BOCPD's delta detector is Storvik-calibrated
         *
         * USAGE:
         *   if (bocpd_changepoint_detected) {
         *       mmpf_inject_shock(mmpf);
         *       mmpf_step(mmpf, obs, &out);
         *       mmpf_restore_from_shock(mmpf);
         *   } else {
         *       mmpf_step(mmpf, obs, &out);
         *   }
         *
         * WARNING:
         *   DO NOT make shock dependent on MMPF state (e.g., outlier fraction).
         *   This creates feedback loops. BOCPD is external watchdog by design.
         *═══════════════════════════════════════════════════════════════════════════*/

        /* Saved transition matrix (restored after shock) */
        rbpf_real_t saved_transition[MMPF_N_MODELS][MMPF_N_MODELS];

        /* Shock state */
        int shock_active;                     /* 1 if currently in shock mode */
        rbpf_real_t process_noise_multiplier; /* Applied to sigma_vol during shock */

        /*═══════════════════════════════════════════════════════════════════════
         * GLOBAL BASELINE TRACKING STATE
         *═══════════════════════════════════════════════════════════════════════*/

        rbpf_real_t global_mu_vol;         /* Current baseline (updated each tick) */
        rbpf_real_t prev_weighted_log_vol; /* Previous output (for EWMA update) */
        int baseline_frozen_ticks;         /* Ticks since baseline was frozen (0 = not frozen) */

        /*═══════════════════════════════════════════════════════════════════════
         * GATED DYNAMICS LEARNING STATE
         *
         * Per-hypothesis sufficient statistics for learning φ and σ_η.
         * These are SEPARATE from Storvik's μ_vol learning (which we disable).
         *═══════════════════════════════════════════════════════════════════════*/

        struct
        {
            double sum_xy;       /* Σ w × x_{t-1} × x_t */
            double sum_xx;       /* Σ w × x_{t-1}² */
            double sum_resid_sq; /* Σ w × (x_t - φ×x_{t-1})² */
            double sum_weight;   /* Σ w (effective sample size) */
            double phi;          /* Current learned φ */
            double sigma_eta;    /* Current learned σ_η */
            double prev_state;   /* x_{t-1} for this hypothesis */
        } gated_dynamics[MMPF_N_MODELS];

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
     * Get underlying RBPF_Extended instance (for advanced access).
     * Each RBPF_Extended contains: RBPF_KSC + Storvik + OCSN
     */
    const RBPF_Extended *mmpf_get_ext(const MMPF_ROCKS *mmpf, MMPF_Hypothesis model);

    /**
     * Get outlier fraction from specific model's OCSN.
     * This is the posterior probability that the last observation
     * was an outlier under that hypothesis's world view.
     */
    rbpf_real_t mmpf_get_model_outlier_fraction(const MMPF_ROCKS *mmpf, MMPF_Hypothesis model);

    /**
     * Get current global baseline (μ_vol).
     * This tracks secular drift in volatility over time.
     */
    rbpf_real_t mmpf_get_global_baseline(const MMPF_ROCKS *mmpf);

    /**
     * Check if baseline is currently frozen (Fair Weather Gate active).
     * Returns 1 if frozen (crisis detected), 0 if updating normally.
     */
    int mmpf_is_baseline_frozen(const MMPF_ROCKS *mmpf);

    /**
     * Get number of ticks baseline has been frozen.
     * Returns 0 if not frozen, >0 if frozen for that many ticks.
     */
    int mmpf_get_baseline_frozen_ticks(const MMPF_ROCKS *mmpf);

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
     * API: BOCPD Shock Mechanism
     *
     * For wiring BOCPD changepoint detection to MMPF. When BOCPD detects a
     * changepoint, call mmpf_inject_shock() before mmpf_step(), then call
     * mmpf_restore_from_shock() after. This forces MMPF to explore all regimes
     * equally for one tick, allowing immediate adaptation to the new regime.
     *
     * Typical usage:
     *
     *   bool changepoint = delta_detector_check(&det, bocpd.r, 3.0);
     *
     *   if (changepoint) {
     *       mmpf_inject_shock(mmpf);
     *       mmpf_step(mmpf, obs, &out);
     *       mmpf_restore_from_shock(mmpf);
     *   } else {
     *       mmpf_step(mmpf, obs, &out);
     *   }
     *
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Inject shock: uniform transitions + boosted process noise.
     *
     * Call this BEFORE mmpf_step() when BOCPD signals a changepoint.
     * Saves current transition matrix for later restoration.
     *
     * Effects:
     *   - Transition matrix → uniform (33% to each regime)
     *   - Process noise (sigma_vol) → multiplied by 50x
     *   - Shock flag set (prevents double-injection)
     *
     * @param mmpf  MMPF instance
     */
    void mmpf_inject_shock(MMPF_ROCKS *mmpf);

    /**
     * Inject shock with custom process noise multiplier.
     *
     * @param mmpf                 MMPF instance
     * @param noise_multiplier     Process noise boost (default: 50.0)
     */
    void mmpf_inject_shock_ex(MMPF_ROCKS *mmpf, rbpf_real_t noise_multiplier);

    /**
     * Restore from shock: reset transitions + process noise.
     *
     * Call this AFTER mmpf_step() when returning to normal operation.
     * Restores the transition matrix that was saved by mmpf_inject_shock().
     *
     * @param mmpf  MMPF instance
     */
    void mmpf_restore_from_shock(MMPF_ROCKS *mmpf);

    /**
     * Check if shock is currently active.
     *
     * @param mmpf  MMPF instance
     * @return      1 if shock active, 0 otherwise
     */
    int mmpf_is_shock_active(const MMPF_ROCKS *mmpf);

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
     * Export particles from RBPF_Extended to buffer.
     */
    void mmpf_buffer_export(MMPF_ParticleBuffer *buf, const RBPF_Extended *ext);

    /**
     * Import particles from buffer to RBPF_Extended.
     */
    void mmpf_buffer_import(const MMPF_ParticleBuffer *buf, RBPF_Extended *ext);

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