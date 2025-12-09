/*
 * ═══════════════════════════════════════════════════════════════════════════
 * RBPF Parameter Learning: Sleeping Storvik + EWSS Comparison
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Primary: "Sleeping Storvik" - Full Bayesian with adaptive sampling frequency
 *   - ALWAYS update sufficient statistics (cheap: O(1) arithmetic)
 *   - CONDITIONALLY sample parameters (expensive: RNG)
 *   - No cold-start problem: stats are always fresh
 *
 * Secondary: EWSS - For comparison/ablation testing only
 *   - Point estimates via online MLE
 *   - No posterior uncertainty
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE "SLEEPING BAYESIAN" INSIGHT
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Traditional approach: Switch between EWSS (fast) and Storvik (accurate)
 *   Problem: Cold start when switching - stats are stale/empty
 *
 * Sleeping Storvik: One algorithm, modulate sampling frequency
 *   - Calm regime (R0/R1): Sample every 50 ticks (parameters frozen)
 *   - Medium regime (R2): Sample every 10 ticks
 *   - Crisis regime (R3): Sample every tick (full adaptation)
 *   - On regime change: Force sample (immediate adaptation)
 *   - On structural break: Force sample (respond to SSA signal)
 *
 * Stats accumulate continuously → first sample after "waking" is correct
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * MODEL
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Log-volatility follows OU/AR(1) per regime:
 *
 *   ℓ_t = μ_r + φ_r(ℓ_{t-1} - μ_r) + σ_r·ε_t,   ε_t ~ N(0,1)
 *
 * Conjugate priors (Normal-Inverse-Gamma):
 *   σ² ~ InvGamma(α, β)
 *   μ | σ² ~ N(m, σ²/κ)
 *
 * Sufficient statistics track posterior exactly for (μ, σ²).
 * φ is fixed per regime (hardest to estimate, least benefit online).
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#ifndef RBPF_PARAM_LEARN_H
#define RBPF_PARAM_LEARN_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /*═══════════════════════════════════════════════════════════════════════════
     * PRECISION & CONSTANTS
     *═══════════════════════════════════════════════════════════════════════════*/

#ifndef PARAM_LEARN_REAL
    typedef double param_real;
#else
typedef PARAM_LEARN_REAL param_real;
#endif

#define PARAM_LEARN_MAX_REGIMES 8
#define PARAM_LEARN_MAX_PARTICLES 1024

    /*═══════════════════════════════════════════════════════════════════════════
     * METHOD SELECTION (Storvik primary, EWSS for comparison)
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef enum
    {
        PARAM_LEARN_SLEEPING_STORVIK, /* Primary: full Bayesian, adaptive sampling */
        PARAM_LEARN_EWSS,             /* Comparison: point estimates only          */
        PARAM_LEARN_FIXED             /* No adaptation: use priors                 */
    } ParamLearnMethod;

    /*═══════════════════════════════════════════════════════════════════════════
     * SAMPLING TRIGGERS (bitfield for diagnostics)
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef enum
    {
        SAMPLE_TRIGGER_NONE = 0,
        SAMPLE_TRIGGER_INTERVAL = 1 << 0,         /* Regular interval elapsed    */
        SAMPLE_TRIGGER_REGIME_CHANGE = 1 << 1,    /* Regime just changed         */
        SAMPLE_TRIGGER_STRUCTURAL_BREAK = 1 << 2, /* SSA detected structure change */
        SAMPLE_TRIGGER_FORCED = 1 << 3,           /* Manual force                */
        SAMPLE_TRIGGER_RESAMPLING = 1 << 4,       /* After particle resampling   */
        SAMPLE_TRIGGER_FIRST = 1 << 5,            /* First time in this regime   */
    } SampleTrigger;

    /*═══════════════════════════════════════════════════════════════════════════
     * PRIOR SPECIFICATION (NIG hyperparameters)
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        /* Normal-Inverse-Gamma prior for (μ, σ²) */
        param_real m;     /* Prior mean for μ                           */
        param_real kappa; /* Prior precision (higher = tighter)         */
        param_real alpha; /* InvGamma shape (≥2 for finite variance)    */
        param_real beta;  /* InvGamma rate                              */

        /* Persistence (fixed per regime) */
        param_real phi; /* AR(1) coefficient / persistence            */

        /* Derived (for convenience) */
        param_real sigma_prior; /* sqrt(β/(α-1)) - prior mean of σ            */

        /*─────────────────────────────────────────────────────────────────────────
         * Precomputed terms (avoid division on hot path)
         *───────────────────────────────────────────────────────────────────────*/
        param_real one_minus_phi;        /* 1 - φ (clamped away from 0)        */
        param_real one_minus_phi_sq;     /* (1 - φ)²                           */
        param_real inv_one_minus_phi;    /* 1 / (1 - φ)                        */
        param_real inv_one_minus_phi_sq; /* 1 / (1 - φ)² = var_scale           */

    } RegimePrior;

/*═══════════════════════════════════════════════════════════════════════════
 * SOA STORAGE: Contiguous arrays for cache efficiency and SIMD
 *
 * Layout: array[particle_idx * n_regimes + regime_idx]
 *
 * Why SoA over AoS:
 *   - Cache: When updating 'm' for all particles, load 8 values per cache line
 *   - SIMD: AVX-512 can process 8 doubles per instruction
 *   - Prefetch: CPU prefetcher predicts sequential access perfectly
 *
 * __restrict tells compiler arrays don't alias → enables vectorization
 *═══════════════════════════════════════════════════════════════════════════*/

/* Memory alignment for AVX-512 */
#define PL_CACHE_LINE 64

/* Force inline for hot path functions */
#if defined(__GNUC__) || defined(__clang__)
#define PL_FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define PL_FORCE_INLINE __forceinline
#else
#define PL_FORCE_INLINE inline
#endif

    typedef struct
    {
        /*─────────────────────────────────────────────────────────────────────────
         * NIG Posterior Hyperparameters (updated every tick - cheap)
         * Aligned arrays: [n_particles * n_regimes], 64-byte aligned
         * __restrict: promise no aliasing → compiler can vectorize
         *───────────────────────────────────────────────────────────────────────*/
        param_real *__restrict m;
        param_real *__restrict kappa;
        param_real *__restrict alpha;
        param_real *__restrict beta;

        /*─────────────────────────────────────────────────────────────────────────
         * Cached Samples (updated only when "awake" - expensive)
         *───────────────────────────────────────────────────────────────────────*/
        param_real *__restrict mu_cached;
        param_real *__restrict sigma2_cached;
        param_real *__restrict sigma_cached;

        /*─────────────────────────────────────────────────────────────────────────
         * Tracking (int arrays)
         *───────────────────────────────────────────────────────────────────────*/
        int *__restrict n_obs;
        int *__restrict ticks_since_sample;

    } StorvikSoA;

    /*═══════════════════════════════════════════════════════════════════════════
     * ENTROPY BUFFER: Pre-generated random numbers (batch RNG)
     *
     * Instead of generating one random number at a time (slow),
     * pre-fill a buffer with MKL VSL or fallback (10x faster).
     *═══════════════════════════════════════════════════════════════════════════*/

#define PL_RNG_BUFFER_SIZE 4096

    typedef struct
    {
        param_real *normal;    /* Pre-generated N(0,1) samples, aligned      */
        param_real *uniform;   /* Pre-generated U(0,1) samples, aligned      */
        int normal_cursor;     /* Current position in normal buffer          */
        int uniform_cursor;    /* Current position in uniform buffer         */
        int buffer_size;       /* Size of each buffer                        */
        uint64_t rng_state[2]; /* xoroshiro128+ state for refill             */
    } EntropyBuffer;

    /*═══════════════════════════════════════════════════════════════════════════
     * EWSS STATISTICS (global per regime, for comparison mode)
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        /* Exponentially-weighted sums for transformed obs z = ℓ - φℓ_lag */
        param_real sum_z;    /* Σ λᵗ w z                               */
        param_real sum_z_sq; /* Σ λᵗ w z²                              */
        param_real eff_n;    /* Σ λᵗ w                                 */

        /* Current MLE estimates */
        param_real mu;
        param_real sigma;

    } EWSSStats;

    /*═══════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        /*─────────────────────────────────────────────────────────────────────────
         * Method Selection
         *───────────────────────────────────────────────────────────────────────*/
        ParamLearnMethod method;

        /*─────────────────────────────────────────────────────────────────────────
         * Sleeping Storvik: Sampling Frequency by Regime
         *
         * sample_interval[r] = N means sample every N ticks in regime r
         * sample_interval[r] = 1 means sample every tick (full Bayesian)
         * sample_interval[r] = 0 means never auto-sample (triggers only)
         *───────────────────────────────────────────────────────────────────────*/
        int sample_interval[PARAM_LEARN_MAX_REGIMES];

        /*─────────────────────────────────────────────────────────────────────────
         * Forced Sampling Triggers
         *───────────────────────────────────────────────────────────────────────*/
        bool sample_on_regime_change;    /* Immediate sample on regime flip */
        bool sample_on_structural_break; /* Immediate sample on SSA break   */
        bool sample_after_resampling;    /* Sample after particle resample  */

        /*─────────────────────────────────────────────────────────────────────────
         * Load-Based Throttling (optional)
         *───────────────────────────────────────────────────────────────────────*/
        bool enable_load_throttling;
        param_real load_skip_threshold; /* Skip sampling if load > this    */

        /*─────────────────────────────────────────────────────────────────────────
         * EWSS Configuration (for comparison mode)
         *───────────────────────────────────────────────────────────────────────*/
        param_real ewss_lambda;    /* Decay factor (0.999)            */
        param_real ewss_min_eff_n; /* Min samples before trusting MLE */

        /*─────────────────────────────────────────────────────────────────────────
         * Constraints (applied during sampling)
         *───────────────────────────────────────────────────────────────────────*/
        param_real sigma_floor_mult; /* σ >= mult * prior_σ (0.1)       */
        param_real sigma_ceil_mult;  /* σ <= mult * prior_σ (5.0)       */
        param_real mu_drift_max;     /* Max |μ - prior_μ| allowed       */

        /*─────────────────────────────────────────────────────────────────────────
         * Prior Strength (effective prior observations)
         *───────────────────────────────────────────────────────────────────────*/
        param_real prior_strength;

        /*─────────────────────────────────────────────────────────────────────────
         * RNG Seed
         *───────────────────────────────────────────────────────────────────────*/
        uint64_t rng_seed;

    } ParamLearnConfig;

    /*═══════════════════════════════════════════════════════════════════════════
     * PARTICLE INFO (input to update)
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        int regime;         /* Current regime assignment                   */
        int prev_regime;    /* Previous regime (for change detection)      */
        param_real ell;     /* Current log-vol estimate E[ℓ_t | y_{1:t}]  */
        param_real ell_lag; /* Previous log-vol estimate                   */
        param_real weight;  /* Particle weight (normalized)                */
    } ParticleInfo;

    /*═══════════════════════════════════════════════════════════════════════════
     * OUTPUT: Parameters for RBPF
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        /* Core parameters */
        param_real mu;     /* Long-run mean                               */
        param_real phi;    /* Persistence (fixed per regime)              */
        param_real sigma;  /* Vol-of-vol (innovation std)                 */
        param_real sigma2; /* σ² for convenience                          */

        /* Posterior uncertainty (Storvik only) */
        param_real mu_post_mean;     /* Posterior mean of μ                     */
        param_real mu_post_std;      /* Posterior std of μ                      */
        param_real sigma2_post_mean; /* Posterior mean of σ²                    */
        param_real sigma2_post_std;  /* Posterior std of σ²                     */

        /* Diagnostics */
        int n_obs;                  /* Observations in this regime                 */
        int ticks_since_sample;     /* How stale is the cached sample?            */
        SampleTrigger last_trigger; /* What caused last sample                 */
        param_real confidence;      /* 0=prior only, 1=data-dominated              */

    } RegimeParams;

    /*═══════════════════════════════════════════════════════════════════════════
     * MAIN LEARNER STRUCTURE
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        /*─────────────────────────────────────────────────────────────────────────
         * Configuration
         *───────────────────────────────────────────────────────────────────────*/
        ParamLearnConfig config;
        int n_regimes;
        int n_particles;

        /*─────────────────────────────────────────────────────────────────────────
         * Priors (one per regime)
         *───────────────────────────────────────────────────────────────────────*/
        RegimePrior priors[PARAM_LEARN_MAX_REGIMES];

        /*─────────────────────────────────────────────────────────────────────────
         * Storvik: SoA Layout for Cache Efficiency and SIMD
         *───────────────────────────────────────────────────────────────────────*/
        StorvikSoA storvik;
        int storvik_total_size; /* n_particles * n_regimes */

        /*─────────────────────────────────────────────────────────────────────────
         * Pre-allocated Scratch Buffer (avoids malloc on hot path)
         *───────────────────────────────────────────────────────────────────────*/
        param_real *resample_scratch; /* For SoA resampling, aligned */

        /*─────────────────────────────────────────────────────────────────────────
         * Entropy Buffer: Pre-generated random numbers (batch RNG)
         *───────────────────────────────────────────────────────────────────────*/
        EntropyBuffer entropy;

        /*─────────────────────────────────────────────────────────────────────────
         * EWSS: Global Per-Regime Stats (for comparison)
         *───────────────────────────────────────────────────────────────────────*/
        EWSSStats ewss[PARAM_LEARN_MAX_REGIMES];

        /*─────────────────────────────────────────────────────────────────────────
         * RNG State (xoroshiro128+)
         *───────────────────────────────────────────────────────────────────────*/
        uint64_t rng[2];

        /*─────────────────────────────────────────────────────────────────────────
         * Runtime State
         *───────────────────────────────────────────────────────────────────────*/
        int tick;                   /* Current tick number             */
        bool structural_break_flag; /* Set by SSA bridge               */
        param_real current_load;    /* System load (0-1)               */

        /*─────────────────────────────────────────────────────────────────────────
         * Diagnostics
         *───────────────────────────────────────────────────────────────────────*/
        uint64_t total_stat_updates;       /* Always increments               */
        uint64_t total_samples_drawn;      /* Only when "awake"               */
        uint64_t samples_skipped_load;     /* Skipped due to load             */
        uint64_t samples_triggered_regime; /* Triggered by regime change      */
        uint64_t samples_triggered_break;  /* Triggered by structural break   */

    } ParamLearner;

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Configuration Presets
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Default: Always-awake Storvik (all regimes sample every tick).
     * Best tracking accuracy across all scenarios including transitions.
     * Latency ~45-50μs with 500 particles.
     */
    ParamLearnConfig param_learn_config_defaults(void);

    /**
     * Sleeping Storvik: calm regimes sleep, crisis awake.
     * Lower latency (~28μs), but poor transition tracking (27-44%).
     * Use only if latency budget < 40μs.
     */
    ParamLearnConfig param_learn_config_sleeping(void);

    /**
     * Always-awake Storvik: sample every tick in all regimes.
     * Most accurate, highest latency.
     */
    ParamLearnConfig param_learn_config_full_bayesian(void);

    /**
     * HFT mode: Aggressive sleeping, sample only on triggers.
     * Lowest latency, relies on regime change / break detection.
     */
    ParamLearnConfig param_learn_config_hft(void);

    /**
     * EWSS mode: For comparison testing only.
     */
    ParamLearnConfig param_learn_config_ewss(void);

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Lifecycle
     *═══════════════════════════════════════════════════════════════════════════*/

    int param_learn_init(ParamLearner *learner,
                         const ParamLearnConfig *config,
                         int n_particles,
                         int n_regimes);

    void param_learn_free(ParamLearner *learner);
    void param_learn_reset(ParamLearner *learner);

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Prior Specification
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Set prior from point estimates (converts to NIG hyperparameters).
     */
    void param_learn_set_prior(ParamLearner *learner,
                               int regime,
                               param_real mu,
                               param_real phi,
                               param_real sigma);

    /**
     * Set prior with explicit NIG hyperparameters.
     */
    void param_learn_set_prior_nig(ParamLearner *learner,
                                   int regime,
                                   param_real m, param_real kappa,
                                   param_real alpha, param_real beta,
                                   param_real phi);

    /**
     * Broadcast priors to all particles (call after setting all priors).
     */
    void param_learn_broadcast_priors(ParamLearner *learner);

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Main Update (call each tick)
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Update all particles for current tick.
     *
     * This is the main hot-path function. It:
     *   1. ALWAYS updates sufficient statistics (cheap)
     *   2. CONDITIONALLY samples parameters based on regime/triggers
     *
     * @param learner     Learner state
     * @param particles   Array of particle info
     * @param n           Number of particles
     */
    void param_learn_update(ParamLearner *learner,
                            const ParticleInfo *particles,
                            int n);

    /**
     * Signal structural break from SSA bridge.
     * Next update will force sampling for affected particles.
     */
    void param_learn_signal_structural_break(ParamLearner *learner);

    /**
     * Set current system load (0-1) for throttling.
     */
    void param_learn_set_load(ParamLearner *learner, param_real load);

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Get Parameters
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Get parameters for a particle's current regime.
     * Returns cached samples (may be stale if sleeping).
     */
    void param_learn_get_params(const ParamLearner *learner,
                                int particle_idx,
                                int regime,
                                RegimeParams *params);

    /**
     * Force immediate resampling for a particle's regime.
     * Use sparingly - defeats the purpose of sleeping.
     */
    void param_learn_force_sample(ParamLearner *learner,
                                  int particle_idx,
                                  int regime);

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Particle Resampling Support
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Copy stats from ancestor after RBPF resampling.
     */
    void param_learn_copy_ancestor(ParamLearner *learner,
                                   int dst_particle,
                                   int src_particle);

    /**
     * Batch copy for systematic resampling.
     * Optionally triggers resampling based on config.
     */
    void param_learn_apply_resampling(ParamLearner *learner,
                                      const int *ancestors,
                                      int n);

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Diagnostics
     *═══════════════════════════════════════════════════════════════════════════*/

    void param_learn_print_summary(const ParamLearner *learner);
    void param_learn_print_regime_stats(const ParamLearner *learner, int regime);

    /**
     * Get aggregate statistics across particles for a regime.
     */
    void param_learn_get_regime_summary(const ParamLearner *learner,
                                        int regime,
                                        param_real *mu_mean,
                                        param_real *mu_std,
                                        param_real *sigma_mean,
                                        param_real *sigma_std,
                                        int *total_obs);

#ifdef __cplusplus
}
#endif

#endif /* RBPF_PARAM_LEARN_H */