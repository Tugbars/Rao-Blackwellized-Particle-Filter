/**
 * @file rbpf_ksc.h
 * @brief RBPF with Kim-Shephard-Chib (1998) log-squared observation model
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * STATE DEFINITION (CRITICAL)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This implementation uses LOG-VOLATILITY as the state variable:
 *
 *   ℓ_t = log(σ_t)    ←── state variable (NOT log-variance!)
 *
 * Many papers (including KSC 1998) use log-VARIANCE h_t = log(σ²_t) = 2·ℓ_t.
 * If importing parameters from such papers, you must convert:
 *
 *   μ_vol (here)  = μ_h / 2       (long-run mean)
 *   σ_vol (here)  = σ_h / 2       (vol-of-vol)
 *   θ (here)      = θ_h           (mean reversion - unchanged)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * OBSERVATION MODEL
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * GAUSSIAN (Original KSC):
 *   Returns:     r_t = σ_t × ε_t = exp(ℓ_t) × ε_t,    ε_t ~ N(0,1)
 *   Transform:   y_t = log(r_t²) = 2ℓ_t + log(ε_t²)
 *   Linear form: y_t = H·ℓ_t + noise,  where H = 2
 *
 *   log(ε_t²) ~ log-χ²(1), approximated as 10-component Gaussian mixture (OCSN).
 *
 * STUDENT-T (Extension):
 *   Returns:     r_t = σ_t × ε_t,    ε_t ~ t_ν(0,1)   ← Heavy tails
 *   Scale mixture: ε_t | λ_t ~ N(0, 1/λ_t),  λ_t ~ Gamma(ν/2, ν/2)
 *   Transform:   y_t = 2ℓ_t + log(ε_t²) = 2ℓ_t + log(χ²/λ_t)
 *              = 2ℓ_t + log(χ²) - log(λ_t)
 *
 *   Given λ_t, this is standard KSC with shifted observation:
 *     y_shifted = y_t + log(λ_t)
 *
 *   Regime-dependent ν enables structural fat-tail discrimination:
 *     Calm:   ν=10 (near-Gaussian, 5σ event ~ 10⁻⁵)
 *     Trend:  ν=5  (moderate tails, 5σ event ~ 10⁻³)
 *     Crisis: ν=3  (heavy tails, 5σ event ~ 10⁻²)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * GPB1 APPROXIMATION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * True posterior is mixture of 10^T Gaussians (exponential growth).
 * We use Generalized Pseudo-Bayesian 1 (GPB1): collapse 10→1 each timestep.
 *
 * Variance calculation uses law of total variance:
 *   Var[X] = E[Var[X|K]] + Var[E[X|K]] = E[X²] - E[X]²
 *
 * Latency target: <15μs for 1000 particles (Gaussian)
 *                 <20μs for 1000 particles (Student-t with λ sampling)
 */

#ifndef RBPF_KSC_H
#define RBPF_KSC_H

#include <mkl.h>
#include <mkl_vsl.h>
#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /*─────────────────────────────────────────────────────────────────────────────
     * CONFIGURATION
     *───────────────────────────────────────────────────────────────────────────*/

#define RBPF_MAX_REGIMES 8
#define RBPF_ALIGN 64
#define RBPF_MAX_THREADS 32
#define RBPF_MAX_SMOOTH_LAG 16 /* Maximum fixed-lag smoothing window */

/* Omori et al. (2007) 10-component mixture for log(χ²(1)) approximation
 * Upgrade from KSC (1998) 7-component: better tail accuracy */
#define KSC_N_COMPONENTS 10

/* Liu-West parameter learning */
#define RBPF_LW_LEARN_MU_VOL 1 /* Learn μ_vol per regime */
#define RBPF_LW_LEARN_SIGMA 0  /* Learn σ_vol per regime (optional) */

    /*─────────────────────────────────────────────────────────────────────────────
     * STUDENT-T CONFIGURATION
     *───────────────────────────────────────────────────────────────────────────*/

/* Default degrees of freedom bounds for ν learning */
#define RBPF_NU_FLOOR 2.5f   /* Below this, variance undefined */
#define RBPF_NU_CEIL 30.0f   /* Above this, essentially Gaussian */
#define RBPF_NU_DEFAULT 5.0f /* Moderate tails */

/* Lambda (auxiliary variable) bounds */
#define RBPF_LAMBDA_FLOOR 0.01f /* Prevent log(λ) → -∞ */
#define RBPF_LAMBDA_CEIL 100.0f /* Prevent numerical issues */

    /*─────────────────────────────────────────────────────────────────────────────
     * PORTABLE SIMD HINTS
     *
     * MSVC requires /openmp:experimental for #pragma omp simd.
     * GCC/Clang/ICC support it with standard OpenMP.
     * We provide RBPF_PRAGMA_SIMD as a portable hint that degrades gracefully.
     *───────────────────────────────────────────────────────────────────────────*/

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
/* MSVC: rely on auto-vectorization with /O2 /arch:AVX2 */
#define RBPF_PRAGMA_SIMD
#elif defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
/* GCC/Clang/ICC: use OpenMP SIMD */
#define RBPF_PRAGMA_SIMD _Pragma("omp simd")
#else
#define RBPF_PRAGMA_SIMD
#endif

    /*─────────────────────────────────────────────────────────────────────────────
     * PRECISION SWITCH
     *
     * Define RBPF_USE_DOUBLE before including this header to use double precision.
     * Default is single precision (float) for HFT latency requirements.
     *
     * Usage:
     *   #define RBPF_USE_DOUBLE
     *   #include "rbpf_ksc.h"
     *
     * Or via CMake:
     *   target_compile_definitions(myapp PRIVATE RBPF_USE_DOUBLE)
     *───────────────────────────────────────────────────────────────────────────*/

#ifdef RBPF_USE_DOUBLE
    typedef double rbpf_real_t;
#define RBPF_REAL_FMT "lf"

/* Math functions */
#define rbpf_exp(x) exp(x)
#define rbpf_log(x) log(x)
#define rbpf_sqrt(x) sqrt(x)
#define rbpf_fabs(x) fabs(x)
#define rbpf_pow(x, y) pow(x, y)
#define rbpf_cos(x) cos(x)
#define rbpf_sin(x) sin(x)
#define rbpf_fmax(x, y) fmax(x, y)

/* MKL VML */
#define rbpf_vsExp vdExp
#define rbpf_vsLn vdLn
#define rbpf_vsSqrt vdSqrt
#define rbpf_vsAdd vdAdd
#define rbpf_vsSub vdSub
#define rbpf_vsMul vdMul
#define rbpf_vsDiv vdDiv
#define rbpf_vsInv vdInv

/* MKL BLAS */
#define rbpf_cblas_asum cblas_dasum
#define rbpf_cblas_scal cblas_dscal
#define rbpf_cblas_dot cblas_ddot
#define rbpf_cblas_axpy cblas_daxpy

/* MKL RNG */
#define RBPF_VSL_RNG_UNIFORM vdRngUniform
#define RBPF_VSL_RNG_GAUSSIAN vdRngGaussian
#define RBPF_VSL_RNG_GAMMA vdRngGamma

/* Constants */
#define RBPF_REAL(x) (x)
#define RBPF_EPS 1e-30
#define RBPF_PI 3.14159265358979323846

#else /* RBPF_USE_FLOAT (default) */
typedef float rbpf_real_t;
#define RBPF_REAL_FMT "f"

/* Math functions */
#define rbpf_exp(x) expf(x)
#define rbpf_log(x) logf(x)
#define rbpf_sqrt(x) sqrtf(x)
#define rbpf_fabs(x) fabsf(x)
#define rbpf_pow(x, y) powf(x, y)
#define rbpf_cos(x) cosf(x)
#define rbpf_sin(x) sinf(x)
#define rbpf_fmax(x, y) fmaxf(x, y)

/* MKL VML */
#define rbpf_vsExp vsExp
#define rbpf_vsLn vsLn
#define rbpf_vsSqrt vsSqrt
#define rbpf_vsAdd vsAdd
#define rbpf_vsSub vsSub
#define rbpf_vsMul vsMul
#define rbpf_vsDiv vsDiv
#define rbpf_vsInv vsInv

/* MKL BLAS */
#define rbpf_cblas_asum cblas_sasum
#define rbpf_cblas_scal cblas_sscal
#define rbpf_cblas_dot cblas_sdot
#define rbpf_cblas_axpy cblas_saxpy

/* MKL RNG */
#define RBPF_VSL_RNG_UNIFORM vsRngUniform
#define RBPF_VSL_RNG_GAUSSIAN vsRngGaussian
#define RBPF_VSL_RNG_GAMMA vsRngGamma

/* Constants */
#define RBPF_REAL(x) (x##f)
#define RBPF_EPS 1e-30f
#define RBPF_PI 3.14159265f

#endif /* RBPF_USE_DOUBLE */

    /*─────────────────────────────────────────────────────────────────────────────
     * PCG32 RNG (same as PF2D for consistency)
     *───────────────────────────────────────────────────────────────────────────*/

    typedef struct
    {
        uint64_t state;
        uint64_t inc;
    } rbpf_pcg32_t;

    static inline void rbpf_pcg32_seed(rbpf_pcg32_t *rng, uint64_t seed, uint64_t seq)
    {
        rng->state = 0U;
        rng->inc = (seq << 1u) | 1u;
        rng->state += seed;
        rng->state = rng->state * 6364136223846793005ULL + rng->inc;
    }

    static inline uint32_t rbpf_pcg32_random(rbpf_pcg32_t *rng)
    {
        uint64_t oldstate = rng->state;
        rng->state = oldstate * 6364136223846793005ULL + rng->inc;
        uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = (uint32_t)(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31));
    }

    static inline rbpf_real_t rbpf_pcg32_uniform(rbpf_pcg32_t *rng)
    {
        return (rbpf_real_t)rbpf_pcg32_random(rng) / RBPF_REAL(4294967296.0);
    }

    /**
     * Fast inverse CDF Gaussian (Acklam's approximation)
     * Much faster than Box-Muller: no log, no cos, just polynomial
     */
    static inline rbpf_real_t rbpf_pcg32_gaussian(rbpf_pcg32_t *rng)
    {
        /* Coefficients for rational approximation */
        static const rbpf_real_t a1 = RBPF_REAL(-3.969683028665376e+01);
        static const rbpf_real_t a2 = RBPF_REAL(2.209460984245205e+02);
        static const rbpf_real_t a3 = RBPF_REAL(-2.759285104469687e+02);
        static const rbpf_real_t a4 = RBPF_REAL(1.383577518672690e+02);
        static const rbpf_real_t a5 = RBPF_REAL(-3.066479806614716e+01);
        static const rbpf_real_t a6 = RBPF_REAL(2.506628277459239e+00);

        static const rbpf_real_t b1 = RBPF_REAL(-5.447609879822406e+01);
        static const rbpf_real_t b2 = RBPF_REAL(1.615858368580409e+02);
        static const rbpf_real_t b3 = RBPF_REAL(-1.556989798598866e+02);
        static const rbpf_real_t b4 = RBPF_REAL(6.680131188771972e+01);
        static const rbpf_real_t b5 = RBPF_REAL(-1.328068155288572e+01);

        static const rbpf_real_t c1 = RBPF_REAL(-7.784894002430293e-03);
        static const rbpf_real_t c2 = RBPF_REAL(-3.223964580411365e-01);
        static const rbpf_real_t c3 = RBPF_REAL(-2.400758277161838e+00);
        static const rbpf_real_t c4 = RBPF_REAL(-2.549732539343734e+00);
        static const rbpf_real_t c5 = RBPF_REAL(4.374664141464968e+00);
        static const rbpf_real_t c6 = RBPF_REAL(2.938163982698783e+00);

        static const rbpf_real_t d1 = RBPF_REAL(7.784695709041462e-03);
        static const rbpf_real_t d2 = RBPF_REAL(3.224671290700398e-01);
        static const rbpf_real_t d3 = RBPF_REAL(2.445134137142996e+00);
        static const rbpf_real_t d4 = RBPF_REAL(3.754408661907416e+00);

        static const rbpf_real_t p_low = RBPF_REAL(0.02425);
        static const rbpf_real_t p_high = RBPF_REAL(1.0) - RBPF_REAL(0.02425);

        rbpf_real_t p = rbpf_pcg32_uniform(rng);
        if (p < RBPF_REAL(1e-10))
            p = RBPF_REAL(1e-10);
        if (p > RBPF_REAL(1.0) - RBPF_REAL(1e-10))
            p = RBPF_REAL(1.0) - RBPF_REAL(1e-10);

        rbpf_real_t q, r;

        if (p < p_low)
        {
            /* Lower tail */
            q = rbpf_sqrt(RBPF_REAL(-2.0) * rbpf_log(p));
            return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                   ((((d1 * q + d2) * q + d3) * q + d4) * q + RBPF_REAL(1.0));
        }
        else if (p <= p_high)
        {
            /* Central region */
            q = p - RBPF_REAL(0.5);
            r = q * q;
            return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
                   (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + RBPF_REAL(1.0));
        }
        else
        {
            /* Upper tail */
            q = rbpf_sqrt(RBPF_REAL(-2.0) * rbpf_log(RBPF_REAL(1.0) - p));
            return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                   ((((d1 * q + d2) * q + d3) * q + d4) * q + RBPF_REAL(1.0));
        }
    }

    /**
     * Gamma sampler using Marsaglia-Tsang method
     * Required for Student-t auxiliary variable sampling
     *
     * @param rng    PCG32 RNG state
     * @param alpha  Shape parameter (must be > 0)
     * @param beta   Rate parameter (must be > 0)
     * @return       Sample from Gamma(alpha, beta)
     */
    static inline rbpf_real_t rbpf_pcg32_gamma(rbpf_pcg32_t *rng, rbpf_real_t alpha, rbpf_real_t beta)
    {
        /* Handle alpha < 1 via Ahrens-Dieter transformation */
        if (alpha < RBPF_REAL(1.0))
        {
            rbpf_real_t u = rbpf_pcg32_uniform(rng);
            return rbpf_pcg32_gamma(rng, alpha + RBPF_REAL(1.0), beta) * rbpf_pow(u, RBPF_REAL(1.0) / alpha);
        }

        /* Marsaglia-Tsang for alpha >= 1 */
        rbpf_real_t d = alpha - RBPF_REAL(1.0) / RBPF_REAL(3.0);
        rbpf_real_t c = RBPF_REAL(1.0) / rbpf_sqrt(RBPF_REAL(9.0) * d);

        for (;;)
        {
            rbpf_real_t x, v;
            do
            {
                x = rbpf_pcg32_gaussian(rng);
                v = RBPF_REAL(1.0) + c * x;
            } while (v <= RBPF_REAL(0.0));

            v = v * v * v;
            rbpf_real_t u = rbpf_pcg32_uniform(rng);
            rbpf_real_t x2 = x * x;

            /* Quick acceptance */
            if (u < RBPF_REAL(1.0) - RBPF_REAL(0.0331) * x2 * x2)
            {
                return d * v / beta;
            }

            /* Slow acceptance */
            if (rbpf_log(u) < RBPF_REAL(0.5) * x2 + d * (RBPF_REAL(1.0) - v + rbpf_log(v)))
            {
                return d * v / beta;
            }
        }
    }

    /*─────────────────────────────────────────────────────────────────────────────
     * STRUCTURES
     *───────────────────────────────────────────────────────────────────────────*/

    typedef struct
    {
        rbpf_real_t theta;     /* Mean reversion speed */
        rbpf_real_t mu_vol;    /* Long-run mean of log-vol */
        rbpf_real_t sigma_vol; /* Vol-of-vol (stored for regularization) */
        rbpf_real_t q;         /* Process variance = sigma_vol² */
    } RBPF_RegimeParams;

    /**
     * Liu-West parameter learning configuration
     */
    typedef struct
    {
        int enabled;               /* 0=off, 1=on */
        rbpf_real_t shrinkage;     /* a in [0.90, 0.99], closer to 1 = slower adaptation */
        rbpf_real_t min_mu_vol;    /* Floor for μ_vol (e.g., log(0.001)) */
        rbpf_real_t max_mu_vol;    /* Ceiling for μ_vol (e.g., log(0.5)) */
        rbpf_real_t min_sigma_vol; /* Floor for σ_vol */
        rbpf_real_t max_sigma_vol; /* Ceiling for σ_vol */
        int learn_mu_vol;          /* Learn μ_vol per regime */
        int learn_sigma_vol;       /* Learn σ_vol per regime */
        int warmup_ticks;          /* Ticks before learning starts */
        int tick_count;            /* Current tick counter */

        /* Aggressive resampling for learning mode */
        rbpf_real_t resample_threshold; /* ESS threshold (0.85 for learning, 0.5 normal) */
        int max_ticks_no_resample;      /* Force resample after this many ticks */
        int ticks_since_resample;       /* Counter */
    } RBPF_LiuWest;

    /**
     * Fixed-lag smoothing history entry
     *
     * Stores aggregate statistics from K ticks ago for smoothed output.
     * Uses O(K × n_regimes) memory, not O(K × n_particles).
     */
    typedef struct
    {
        rbpf_real_t vol_mean;                       /* E[exp(ℓ)] at this tick */
        rbpf_real_t log_vol_mean;                   /* E[ℓ] at this tick */
        rbpf_real_t log_vol_var;                    /* Var[ℓ] at this tick */
        rbpf_real_t regime_probs[RBPF_MAX_REGIMES]; /* p(r) at this tick */
        int dominant_regime;                        /* argmax p(r) at this tick */
        rbpf_real_t ess;                            /* ESS at this tick */
        int valid;                                  /* 1 if entry is filled */
    } RBPF_SmoothEntry;

    /**
     * Self-aware detection state (no external model needed)
     */
    typedef struct
    {
        rbpf_real_t vol_ema_short; /* Fast EMA of vol_mean */
        rbpf_real_t vol_ema_long;  /* Slow EMA of vol_mean */
        int prev_regime;           /* For structural change detection */
        int cooldown;              /* Ticks since last detection */

        /* Regime smoothing (hysteresis to prevent flickering) */
        int stable_regime;          /* Smoothed regime output */
        int candidate_regime;       /* Current candidate for new stable regime */
        int hold_count;             /* Ticks candidate has been dominant */
        int hold_threshold;         /* Ticks required before switching (default: 5) */
        rbpf_real_t prob_threshold; /* Probability for immediate switch (default: 0.7) */

        /* Confirmation window (reduces false positives) */
        int consecutive_surprise;  /* Consecutive ticks with high surprise */
        int consecutive_vol_spike; /* Consecutive ticks with high vol ratio */
        int confirm_minor;         /* Required consecutive for minor alert (default: 2) */
        int confirm_major;         /* Required consecutive for major alert (default: 2) */
    } RBPF_Detection;

    /*═══════════════════════════════════════════════════════════════════════════
     * ROBUST OCSN (11TH COMPONENT)
     *
     * Adds outlier-handling to the 10-component OCSN mixture.
     * P(obs) = (1 - π_outlier) × P_OCSN(obs) + π_outlier × P_broad(obs)
     *
     * Prevents particle collapse during 8σ+ moves.
     *
     * VARIANCE BOUNDS (Critical for Signal Preservation):
     *   Max OCSN variance = 7.33
     *   Recommended outlier variance = 2-4× max OCSN ≈ 15-30
     *
     *   Too low (<15):  Outlier competes with OCSN, distorts normal updates
     *   Too high (>50): Kalman gain → 0, signal suppression (ignores crash)
     *   Sweet spot:     ~22 (3× OCSN max) for most assets
     *═══════════════════════════════════════════════════════════════════════════*/

/* Variance bounds for safety */
#define RBPF_OUTLIER_VAR_MIN RBPF_REAL(12.0)     /* 1.6× max OCSN */
#define RBPF_OUTLIER_VAR_MAX RBPF_REAL(50.0)     /* 7× max OCSN */
#define RBPF_OUTLIER_VAR_DEFAULT RBPF_REAL(22.0) /* 3× max OCSN - sweet spot */

    typedef struct
    {
        rbpf_real_t prob;     /* Outlier probability (e.g., 0.01 = 1%) */
        rbpf_real_t variance; /* Outlier variance (recommended: 15-35) */
    } RBPF_OutlierParams;

    typedef struct
    {
        int enabled;
        RBPF_OutlierParams regime[RBPF_MAX_REGIMES]; /* Per-regime params */
    } RBPF_RobustOCSN;

    /*═══════════════════════════════════════════════════════════════════════════
     * STUDENT-T OBSERVATION MODEL CONFIGURATION
     *
     * Replaces Gaussian ε ~ N(0,1) with Student-t ε ~ t_ν(0,1)
     *
     * Key insight: t_ν is a scale mixture of Gaussians:
     *   ε | λ ~ N(0, 1/λ),  λ ~ Gamma(ν/2, ν/2)
     *
     * Benefits:
     *   - Fat tails are STRUCTURAL, not anomalies
     *   - Regime-dependent ν enables natural discrimination
     *   - Crisis with ν=3 expects fat tails → wins likelihood during crashes
     *
     * Implementation:
     *   1. Sample λ from conditional posterior given observation
     *   2. Run standard KSC update with shifted observation y + log(λ)
     *   3. Accumulate λ statistics for optional ν learning
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        int enabled;          /* 0=Gaussian, 1=Student-t */
        rbpf_real_t nu;       /* Degrees of freedom (default: 5) */
        rbpf_real_t nu_floor; /* Minimum ν (default: 2.5) */
        rbpf_real_t nu_ceil;  /* Maximum ν (default: 30) */

        /* ν learning (optional - usually ν is structural/fixed) */
        int learn_nu;                 /* 0=fixed, 1=learn from data */
        rbpf_real_t nu_learning_rate; /* EWMA rate for ν estimation */
    } RBPF_StudentT_Config;

    /**
     * Student-t sufficient statistics for ν learning
     *
     * Uses method of moments on sampled λ values:
     *   E[λ] = 1,  Var[λ] = 2/ν  →  ν = 2/Var[λ]
     *
     * Or digamma method (more accurate):
     *   E[log(λ)] = ψ(ν/2) - log(ν/2)
     */
    typedef struct
    {
        rbpf_real_t sum_lambda;     /* Σ λ (with forgetting) */
        rbpf_real_t sum_lambda_sq;  /* Σ λ² (for variance) */
        rbpf_real_t sum_log_lambda; /* Σ log(λ) (for digamma method) */
        rbpf_real_t n_eff;          /* Effective sample count */
        rbpf_real_t nu_estimate;    /* Current ν estimate */
    } RBPF_StudentT_Stats;

    /**
     * Main RBPF-KSC structure
     */
    typedef struct
    {
        int n_particles;
        int n_regimes;

        /*========================================================================
         * PARTICLE STATE (SoA layout)
         *======================================================================*/
        rbpf_real_t *mu;         /* [n] log-vol mean (Kalman state) */
        rbpf_real_t *var;        /* [n] log-vol variance (Kalman covariance) */
        int *regime;             /* [n] regime index */
        rbpf_real_t *log_weight; /* [n] log-weights for numerical stability */

        /* Double buffers for resampling (pointer swap, no memcpy) */
        rbpf_real_t *mu_tmp;
        rbpf_real_t *var_tmp;
        int *regime_tmp;

        /* Alias for compatibility: some code uses weight instead of log_weight */
        rbpf_real_t *weight; /* Points to w_norm after normalization */

        /*========================================================================
         * REGIME SYSTEM
         *======================================================================*/
        RBPF_RegimeParams params[RBPF_MAX_REGIMES];
        uint8_t trans_lut[RBPF_MAX_REGIMES][1024]; /* Precomputed transition LUT */

        /*========================================================================
         * WORKSPACE (preallocated - NO malloc in hot path)
         *======================================================================*/
        rbpf_real_t *mu_pred;    /* [n] predicted mean */
        rbpf_real_t *var_pred;   /* [n] predicted variance */
        rbpf_real_t *theta_arr;  /* [n] gathered theta per particle */
        rbpf_real_t *mu_vol_arr; /* [n] gathered mu_vol per particle */
        rbpf_real_t *q_arr;      /* [n] gathered q per particle */
        rbpf_real_t *lik_total;  /* [n] total likelihood per particle */
        rbpf_real_t *lik_comp;   /* [n] likelihood for current component */
        rbpf_real_t *innov;      /* [n] innovation */
        rbpf_real_t *S;          /* [n] innovation variance */
        rbpf_real_t *K;          /* [n] Kalman gain */
        rbpf_real_t *w_norm;     /* [n] normalized weights */
        rbpf_real_t *cumsum;     /* [n] cumulative sum for resampling */
        rbpf_real_t *mu_accum;   /* [n] accumulated mu across mixture */
        rbpf_real_t *var_accum;  /* [n] accumulated var across mixture */
        rbpf_real_t *scratch1;   /* [n] general workspace */
        rbpf_real_t *scratch2;   /* [n] general workspace */
        int *indices;            /* [n] resampling indices */

        /* Log-sum-exp buffer for K-mixture (numerical stability) */
        rbpf_real_t *log_lik_buffer; /* [KSC_N_COMPONENTS * n] log-lik per component per particle */
        rbpf_real_t *max_log_lik;    /* [n] max log-lik across components */

        /* Pre-generated Gaussian randoms (MKL ICDF) */
        rbpf_real_t *rng_gaussian; /* [2*n] pre-generated N(0,1) for jitter */
        int rng_buffer_size;       /* Current buffer size */

        /*========================================================================
         * STUDENT-T AUXILIARY VARIABLES
         *======================================================================*/
        rbpf_real_t *lambda;     /* [n] Sampled scale variables (1/precision) */
        rbpf_real_t *lambda_tmp; /* [n] Double buffer for resampling */
        rbpf_real_t *log_lambda; /* [n] log(λ) for shifted observation */

        /* Per-regime Student-t configuration */
        RBPF_StudentT_Config student_t[RBPF_MAX_REGIMES];

        /* Per-regime ν learning statistics */
        RBPF_StudentT_Stats student_t_stats[RBPF_MAX_REGIMES];

        /* Global Student-t enable flag */
        int student_t_enabled;

        /*========================================================================
         * RNG
         *======================================================================*/
        rbpf_pcg32_t pcg[RBPF_MAX_THREADS];
        VSLStreamStatePtr mkl_rng[RBPF_MAX_THREADS];
        int n_threads;

        /*========================================================================
         * DETECTION STATE
         *======================================================================*/
        RBPF_Detection detection;

        /*========================================================================
         * REGULARIZATION
         *======================================================================*/
        rbpf_real_t reg_bandwidth_mu;  /* Jitter bandwidth for mu after resample */
        rbpf_real_t reg_bandwidth_var; /* Jitter bandwidth for var after resample */
        rbpf_real_t reg_scale_min;
        rbpf_real_t reg_scale_max;
        rbpf_real_t last_ess;

        /* Regime diversity preservation (prevents particle collapse to single regime) */
        int min_particles_per_regime;     /* Minimum particles guaranteed per regime */
        rbpf_real_t regime_mutation_prob; /* Probability of random regime mutation [0, 0.1] */

        /*========================================================================
         * LIU-WEST PARAMETER LEARNING
         *======================================================================*/
        RBPF_LiuWest liu_west;

        /* Per-particle parameters (learned online) */
        rbpf_real_t *particle_mu_vol;     /* [n * n_regimes] μ_vol per particle per regime */
        rbpf_real_t *particle_sigma_vol;  /* [n * n_regimes] σ_vol per particle per regime */
        rbpf_real_t *particle_mu_vol_tmp; /* Double buffer */
        rbpf_real_t *particle_sigma_vol_tmp;

        /* Liu-West workspace */
        rbpf_real_t lw_mu_vol_mean[RBPF_MAX_REGIMES];    /* Weighted mean of μ_vol */
        rbpf_real_t lw_mu_vol_var[RBPF_MAX_REGIMES];     /* Weighted variance of μ_vol */
        rbpf_real_t lw_sigma_vol_mean[RBPF_MAX_REGIMES]; /* Weighted mean of σ_vol */
        rbpf_real_t lw_sigma_vol_var[RBPF_MAX_REGIMES];  /* Weighted variance of σ_vol */

        /*========================================================================
         * FIXED-LAG SMOOTHING (Dual Output)
         *
         * Stores history for K-lag smoothed estimates:
         *   - Fast signal (t):   Immediate filtered estimate for rapid reaction
         *   - Smooth signal (t-K): Delayed estimate for regime confirmation
         *======================================================================*/
        int smooth_lag;                                       /* K = smoothing lag (0 = disabled) */
        int smooth_head;                                      /* Circular buffer head index */
        int smooth_count;                                     /* Number of valid entries */
        RBPF_SmoothEntry smooth_history[RBPF_MAX_SMOOTH_LAG]; /* Circular buffer */

        /*========================================================================
         * PRECOMPUTED
         *======================================================================*/
        rbpf_real_t uniform_weight; /* 1/n */
        rbpf_real_t inv_n;

        int use_learned_params; /* 1 = predict reads particle_mu/sigma_vol arrays */

    } RBPF_KSC;

    /**
     * Output from rbpf_ksc_step()
     */
    typedef struct
    {
        /*========================================================================
         * FAST SIGNAL (t) - for immediate "duck and cover" reactions
         *======================================================================*/
        rbpf_real_t vol_mean;     /* E[exp(ℓ)] */
        rbpf_real_t log_vol_mean; /* E[ℓ] */
        rbpf_real_t log_vol_var;  /* Var[ℓ] (includes Kalman uncertainty) */
        rbpf_real_t ess;          /* Effective sample size */

        /* Regime (fast) */
        rbpf_real_t regime_probs[RBPF_MAX_REGIMES];
        int dominant_regime; /* Instantaneous dominant regime */
        int smoothed_regime; /* Smoothed regime (with hysteresis) */

        /* Self-aware signals (Phase 1) */
        rbpf_real_t marginal_lik;   /* p(y_t | y_{1:t-1}) - EXACT from Kalman */
        rbpf_real_t surprise;       /* -log(marginal_lik) */
        rbpf_real_t vol_ratio;      /* vol_mean / vol_ema */
        rbpf_real_t regime_entropy; /* -Σ p·log(p) */

        /* Detection flags */
        int regime_changed; /* 0 or 1 */
        int change_type;    /* 0=none, 1=structural, 2=vol_shock, 3=surprise */

        /*========================================================================
         * SMOOTH SIGNAL (t-K) - for "state of the world" adjustments
         *
         * Fixed-lag smoothed estimates with K-tick delay.
         * Use for: position sizing, spread adjustment, regime confirmation.
         * Valid only when smooth_valid == 1 (after K ticks of warmup).
         *======================================================================*/
        int smooth_valid; /* 1 if smooth signals are valid (after K ticks) */
        int smooth_lag;   /* K = current smoothing lag */

        rbpf_real_t vol_mean_smooth;     /* E[exp(ℓ)] at t-K */
        rbpf_real_t log_vol_mean_smooth; /* E[ℓ] at t-K */
        rbpf_real_t log_vol_var_smooth;  /* Var[ℓ] at t-K */

        rbpf_real_t regime_probs_smooth[RBPF_MAX_REGIMES]; /* p(r) at t-K */
        int dominant_regime_smooth;                        /* argmax p(r) at t-K */
        rbpf_real_t regime_confidence;                     /* max(regime_probs_smooth) - how sure are we? */

        /*========================================================================
         * LEARNED PARAMETERS & DIAGNOSTICS
         *======================================================================*/
        /* Liu-West learned parameters (Phase 3) */
        rbpf_real_t learned_mu_vol[RBPF_MAX_REGIMES];    /* Weighted mean of μ_vol per regime */
        rbpf_real_t learned_sigma_vol[RBPF_MAX_REGIMES]; /* Weighted mean of σ_vol per regime */

        /* Diagnostics */
        int resampled;
        int apf_triggered; /* 1 if APF lookahead was used this step */

        /* Robust OCSN diagnostics */
        rbpf_real_t outlier_fraction; /* Weighted avg P(outlier | obs), [0,1] */

        /*========================================================================
         * STUDENT-T DIAGNOSTICS
         *======================================================================*/
        rbpf_real_t lambda_mean;                  /* E[λ] across particles */
        rbpf_real_t lambda_var;                   /* Var[λ] across particles */
        rbpf_real_t nu_effective;                 /* Implied ν from λ variance: ν = 2/Var[λ] */
        rbpf_real_t learned_nu[RBPF_MAX_REGIMES]; /* Current ν estimate per regime */
        int student_t_active;                     /* 1 if Student-t update was used */

    } RBPF_KSC_Output;

    /*─────────────────────────────────────────────────────────────────────────────
     * API
     *───────────────────────────────────────────────────────────────────────────*/

    /* Create/destroy */
    RBPF_KSC *rbpf_ksc_create(int n_particles, int n_regimes);
    void rbpf_ksc_destroy(RBPF_KSC *rbpf);

    /* Configuration */
    void rbpf_ksc_set_regime_params(RBPF_KSC *rbpf, int r,
                                    rbpf_real_t theta, rbpf_real_t mu_vol, rbpf_real_t sigma_vol);
    void rbpf_ksc_build_transition_lut(RBPF_KSC *rbpf, const rbpf_real_t *trans_matrix);
    void rbpf_ksc_set_regularization(RBPF_KSC *rbpf, rbpf_real_t h_mu, rbpf_real_t h_var);

    /* Regime diversity: prevent particle collapse to single regime
     * min_per_regime: minimum particles per regime (0 to disable)
     * mutation_prob: probability of random regime mutation [0, 0.1] */
    void rbpf_ksc_set_regime_diversity(RBPF_KSC *rbpf, int min_per_regime, rbpf_real_t mutation_prob);

    /* Regime smoothing: prevent regime flickering with hysteresis
     * hold_threshold: ticks new regime must hold before switching (default: 5)
     * prob_threshold: probability for immediate switch (default: 0.7) */
    void rbpf_ksc_set_regime_smoothing(RBPF_KSC *rbpf, int hold_threshold, rbpf_real_t prob_threshold);

    /* Fixed-lag smoothing: dual output for HFT
     * lag: number of ticks delay for smooth signal (0 to disable, max RBPF_MAX_SMOOTH_LAG)
     *
     * After enabling, output contains:
     *   - Fast signal (t):   vol_mean, dominant_regime, surprise (immediate reaction)
     *   - Smooth signal (t-K): vol_mean_smooth, dominant_regime_smooth (regime confirmation)
     *
     * Typical usage: lag=5 for regime confirmation, lag=0 for pure filtering */
    void rbpf_ksc_set_fixed_lag_smoothing(RBPF_KSC *rbpf, int lag);

    /* Liu-West parameter learning (Phase 3) */
    void rbpf_ksc_enable_liu_west(RBPF_KSC *rbpf, rbpf_real_t shrinkage, int warmup_ticks);
    void rbpf_ksc_disable_liu_west(RBPF_KSC *rbpf);
    void rbpf_ksc_set_liu_west_bounds(RBPF_KSC *rbpf,
                                      rbpf_real_t min_mu_vol, rbpf_real_t max_mu_vol,
                                      rbpf_real_t min_sigma_vol, rbpf_real_t max_sigma_vol);
    void rbpf_ksc_set_liu_west_resample(RBPF_KSC *rbpf,
                                        rbpf_real_t ess_threshold,  /* 0.85 = resample at 85% ESS */
                                        int max_ticks_no_resample); /* Force resample after N ticks */
    void rbpf_ksc_get_learned_params(const RBPF_KSC *rbpf, int regime,
                                     rbpf_real_t *mu_vol_out, rbpf_real_t *sigma_vol_out);

    /* PMMH injection (call every 50-100 ticks with offline PMMH results)
     * Resets per-particle parameters toward PMMH estimates with controlled jitter */
    void rbpf_ksc_inject_pmmh(RBPF_KSC *rbpf, int regime,
                              rbpf_real_t pmmh_mu_vol, rbpf_real_t pmmh_sigma_vol,
                              rbpf_real_t blend); /* blend in [0,1]: 0=keep, 1=full reset */

    /* Inject all regimes at once */
    void rbpf_ksc_inject_pmmh_all(RBPF_KSC *rbpf,
                                  const rbpf_real_t *pmmh_mu_vol,    /* [n_regimes] */
                                  const rbpf_real_t *pmmh_sigma_vol, /* [n_regimes] */
                                  rbpf_real_t blend);

    /*─────────────────────────────────────────────────────────────────────────────
     * STUDENT-T API
     *
     * Student-t observation model for fat-tail robustness.
     *
     * Two usage modes:
     *   1. FIXED ν (structural): Set once, different ν per hypothesis in MMPF
     *   2. LEARNED ν (adaptive): Storvik-style learning from λ statistics
     *
     * Recommended: Fixed ν for MMPF (ν IS the hypothesis identity)
     *───────────────────────────────────────────────────────────────────────────*/

    /**
     * Enable Student-t observation model globally
     *
     * @param rbpf     RBPF instance
     * @param nu       Default degrees of freedom (applied to all regimes)
     */
    void rbpf_ksc_enable_student_t(RBPF_KSC *rbpf, rbpf_real_t nu);

    /**
     * Disable Student-t, revert to Gaussian observations
     */
    void rbpf_ksc_disable_student_t(RBPF_KSC *rbpf);

    /**
     * Set regime-specific ν (for MMPF per-hypothesis configuration)
     *
     * @param rbpf     RBPF instance
     * @param regime   Regime index
     * @param nu       Degrees of freedom for this regime
     */
    void rbpf_ksc_set_student_t_nu(RBPF_KSC *rbpf, int regime, rbpf_real_t nu);

    /**
     * Enable ν learning for a regime
     *
     * Uses accumulated λ statistics to estimate ν online.
     * Typically NOT recommended for MMPF where ν should be structural.
     *
     * @param rbpf           RBPF instance
     * @param regime         Regime index
     * @param learning_rate  EWMA decay for λ statistics (0.99 = slow, 0.95 = fast)
     */
    void rbpf_ksc_enable_nu_learning(RBPF_KSC *rbpf, int regime, rbpf_real_t learning_rate);

    /**
     * Disable ν learning, fix to current estimate
     */
    void rbpf_ksc_disable_nu_learning(RBPF_KSC *rbpf, int regime);

    /**
     * Get current ν estimate for regime
     */
    rbpf_real_t rbpf_ksc_get_nu(const RBPF_KSC *rbpf, int regime);

    /**
     * Get λ statistics for regime (for diagnostics/Storvik integration)
     */
    void rbpf_ksc_get_lambda_stats(const RBPF_KSC *rbpf, int regime,
                                   rbpf_real_t *mean_out, rbpf_real_t *var_out,
                                   rbpf_real_t *n_eff_out);

    /**
     * Reset ν learning statistics
     */
    void rbpf_ksc_reset_nu_learning(RBPF_KSC *rbpf, int regime);

    /*─────────────────────────────────────────────────────────────────────────────
     * STUDENT-T UPDATE FUNCTIONS
     *───────────────────────────────────────────────────────────────────────────*/

    /**
     * Student-t Kalman update via auxiliary variable λ
     *
     * Algorithm:
     *   1. For each particle, sample λ from conditional posterior
     *   2. Shift observation: y_shifted = y + log(λ)
     *   3. Run standard 10-component KSC update with y_shifted
     *   4. Accumulate λ statistics for ν learning (if enabled)
     *
     * @param rbpf   RBPF instance (must have Student-t enabled)
     * @param y      Transformed observation: y = log(r²)
     * @return       Marginal likelihood p(y_t | y_{1:t-1})
     */
    rbpf_real_t rbpf_ksc_update_student_t(RBPF_KSC *rbpf, rbpf_real_t y);

    /**
     * Student-t update with explicit ν (ignores per-regime settings)
     *
     * Useful for MMPF where each hypothesis has its own RBPF and ν.
     *
     * @param rbpf   RBPF instance
     * @param y      Transformed observation: y = log(r²)
     * @param nu     Degrees of freedom to use
     * @return       Marginal likelihood
     */
    rbpf_real_t rbpf_ksc_update_student_t_nu(RBPF_KSC *rbpf, rbpf_real_t y, rbpf_real_t nu);

    /**
     * Combined step with Student-t observations
     *
     * Equivalent to rbpf_ksc_step() but uses Student-t update internally.
     *
     * @param rbpf    RBPF instance
     * @param obs     Raw observation (return r_t, NOT log(r²))
     * @param output  Output structure
     */
    void rbpf_ksc_step_student_t(RBPF_KSC *rbpf, rbpf_real_t obs, RBPF_KSC_Output *output);

    /**
     * Combined step with explicit ν
     */
    void rbpf_ksc_step_student_t_nu(RBPF_KSC *rbpf, rbpf_real_t obs, rbpf_real_t nu,
                                    RBPF_KSC_Output *output);

    /*─────────────────────────────────────────────────────────────────────────────
     * STANDARD API (continued)
     *───────────────────────────────────────────────────────────────────────────*/

    /* Initialize */
    void rbpf_ksc_init(RBPF_KSC *rbpf, rbpf_real_t mu0, rbpf_real_t var0);

    /* Main update - THE HOT PATH */
    void rbpf_ksc_step(RBPF_KSC *rbpf, rbpf_real_t obs, RBPF_KSC_Output *output);

    /* Warmup (call once before trading) */
    void rbpf_ksc_warmup(RBPF_KSC *rbpf);

    /* Debug */
    void rbpf_ksc_print_config(const RBPF_KSC *rbpf);

    /*─────────────────────────────────────────────────────────────────────────────
     * Internal functions (exposed for APF extension)
     *
     * These are the building blocks of rbpf_ksc_step(). They're exposed so that
     * rbpf_apf.c can construct alternative step functions (e.g., with lookahead).
     *───────────────────────────────────────────────────────────────────────────*/

    void rbpf_ksc_predict(RBPF_KSC *rbpf);
    rbpf_real_t rbpf_ksc_update(RBPF_KSC *rbpf, rbpf_real_t y);
    void rbpf_ksc_transition(RBPF_KSC *rbpf);
    int rbpf_ksc_resample(RBPF_KSC *rbpf);
    void rbpf_ksc_compute_outputs(RBPF_KSC *rbpf, rbpf_real_t marginal_lik,
                                  RBPF_KSC_Output *out);

    /*─────────────────────────────────────────────────────────────────────────────
     * ROBUST OCSN UPDATE (11th Component)
     *
     * Modified Kalman update that adds an outlier component to the mixture.
     * Prevents particle collapse during extreme observations (8σ+ moves).
     *
     * @param rbpf        RBPF context
     * @param y           Transformed observation: y = log(r²)
     * @param robust_ocsn Robust OCSN configuration (per-regime outlier params)
     * @return            Marginal likelihood p(y_t | y_{1:t-1})
     *───────────────────────────────────────────────────────────────────────────*/
    rbpf_real_t rbpf_ksc_update_robust(RBPF_KSC *rbpf, rbpf_real_t y,
                                       const RBPF_RobustOCSN *robust_ocsn);

    /**
     * Compute outlier fraction for diagnostics
     *
     * Returns the fraction of likelihood explained by the outlier component.
     * High values (>0.5) indicate the observation was treated as an outlier.
     *
     * @param rbpf        RBPF context
     * @param y           Transformed observation: y = log(r²)
     * @param robust_ocsn Robust OCSN configuration
     * @return            Outlier fraction [0, 1]
     */
    rbpf_real_t rbpf_ksc_compute_outlier_fraction(const RBPF_KSC *rbpf,
                                                  rbpf_real_t y,
                                                  const RBPF_RobustOCSN *robust_ocsn);

    /*─────────────────────────────────────────────────────────────────────────────
     * APF (Auxiliary Particle Filter) API
     *
     * Lookahead-based resampling for improved regime change detection.
     * Requires r_{t+1} to be available (1-tick lookahead).
     *
     * ═══════════════════════════════════════════════════════════════════════════
     * SPLIT-STREAM ARCHITECTURE (CRITICAL)
     * ═══════════════════════════════════════════════════════════════════════════
     *
     * For best results, use DIFFERENT data streams for update vs lookahead:
     *
     *   obs_current: SSA-CLEANED return → stable state update
     *   obs_next:    RAW tick return    → see full spike for lookahead
     *
     * Why: If you smooth the lookahead (SSA on obs_next), you remove the
     * "surprise" signal that triggers aggressive APF resampling. SSA-smoothed
     * lookahead turns a "crisis detector" into a "laggy tracker".
     *
     * Example usage:
     *   rbpf_ksc_step_apf(rbpf, ssa_return[t], raw_return[t+1], &output);
     *
     * Key improvements in this implementation:
     *   1. Variance inflation (2.5x): Widen search beam for 5σ spikes
     *   2. Shotgun sampling: Evaluate at mean, ±2σ, take best
     *   3. Mixture proposal (α=0.8): 80% APF + 20% SIR for diversity
     *───────────────────────────────────────────────────────────────────────────*/

    /* Full APF step - always uses lookahead (~20μs with shotgun sampling)
     * obs_current: SSA-cleaned return for UPDATE (stable estimate)
     * obs_next:    RAW return for LOOKAHEAD (see the spike) */
    void rbpf_ksc_step_apf(RBPF_KSC *rbpf, rbpf_real_t obs_current, rbpf_real_t obs_next,
                           RBPF_KSC_Output *output);

    /* Adaptive APF - switches based on surprise level (12-20μs)
     * Uses standard SIR in calm markets, APF during regime changes */
    void rbpf_ksc_step_adaptive(RBPF_KSC *rbpf, rbpf_real_t obs_current, rbpf_real_t obs_next,
                                RBPF_KSC_Output *output);

    /* Force APF for next n_steps (call when BOCPD signals changepoint) */
    void rbpf_ksc_force_apf(int n_steps);
    int rbpf_ksc_apf_forced(void);

    /* APF statistics */
    void rbpf_apf_reset_stats(void);
    void rbpf_apf_get_stats(int *total, int *apf_count, rbpf_real_t *apf_ratio);

    /*─────────────────────────────────────────────────────────────────────────────
     * INTERNAL FUNCTIONS (exposed for APF module)
     *───────────────────────────────────────────────────────────────────────────*/
    void rbpf_ksc_predict_internal(RBPF_KSC *rbpf);
    rbpf_real_t rbpf_ksc_update_internal(RBPF_KSC *rbpf, rbpf_real_t y);
    void rbpf_ksc_resample_internal(RBPF_KSC *rbpf);
    void rbpf_ksc_transition_internal(RBPF_KSC *rbpf);
    void rbpf_ksc_compute_outputs_internal(RBPF_KSC *rbpf, rbpf_real_t marginal, RBPF_KSC_Output *out);
    void rbpf_ksc_liu_west_update_internal(RBPF_KSC *rbpf);

    /*═══════════════════════════════════════════════════════════════════════════
     * RBPF PIPELINE: Unified Change Detection + Volatility Tracking
     *
     * Replaces BOCPD + PF stack with single filter.
     *
     * Stack: SSA → RBPF Pipeline → Kelly
     *
     * Provides:
     *   - Volatility forecast (vol_forecast) for Kelly bet sizing
     *   - Regime identification (regime) for parameter selection
     *   - Change detection (change_detected) for risk management
     *   - Position scaling (position_scale) as direct Kelly multiplier
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Pipeline signal - everything Kelly needs in one struct
     */
    typedef struct
    {
        /* Volatility for Kelly */
        rbpf_real_t vol_forecast;    /* E[σ] for next tick */
        rbpf_real_t vol_uncertainty; /* Std[σ] for conservative sizing */
        rbpf_real_t log_vol;         /* E[log(σ)] */

        /* Regime for parameter selection */
        int regime;                    /* 0=calm, 1=normal, 2=elevated, 3=crisis */
        rbpf_real_t regime_confidence; /* P(regime correct) */
        rbpf_real_t regime_probs[RBPF_MAX_REGIMES];

        /* Change detection for risk management */
        int change_detected;   /* 0=none, 1=minor, 2=major */
        int change_type;       /* 0=none, 1=vol_spike, 2=regime_shift, 3=both */
        rbpf_real_t surprise;  /* -log(p(obs)) */
        rbpf_real_t vol_ratio; /* short_ema / long_ema */

        /* Direct Kelly multiplier */
        rbpf_real_t position_scale; /* 0.0-1.0, multiply Kelly fraction by this */
        int action;                 /* 0=normal, 1=reduce, 2=exit */

        /* Diagnostics */
        rbpf_real_t ess;
        rbpf_real_t regime_entropy;
        int tick;
    } RBPF_Signal;

    /**
     * Pipeline configuration
     */
    typedef struct
    {
        int n_particles;
        int n_regimes;

        rbpf_real_t theta[RBPF_MAX_REGIMES];
        rbpf_real_t mu_vol[RBPF_MAX_REGIMES];
        rbpf_real_t sigma_vol[RBPF_MAX_REGIMES];
        rbpf_real_t transition[RBPF_MAX_REGIMES * RBPF_MAX_REGIMES];

        rbpf_real_t surprise_minor, surprise_major;
        rbpf_real_t surprise_extreme; /* Extreme threshold - bypasses confirmation */
        rbpf_real_t vol_ratio_minor, vol_ratio_major;
        rbpf_real_t scale_on_minor, scale_on_major;
        rbpf_real_t scale_low_confidence, confidence_threshold;

        /* Confirmation window (reduces false positives) */
        int confirm_minor; /* Consecutive ticks for minor alert (default: 2) */
        int confirm_major; /* Consecutive ticks for major alert (default: 2) */

        /* Outlier protection */
        rbpf_real_t outlier_clip_sigma; /* Clip observations beyond N sigma (default: 8.0) */

        int smooth_lag, regime_hold_ticks;
        int enable_learning;
        rbpf_real_t learning_shrinkage;
        int learning_warmup;

        /*========================================================================
         * STUDENT-T PIPELINE CONFIGURATION
         *======================================================================*/
        int enable_student_t;                       /* Use Student-t observations */
        rbpf_real_t student_t_nu[RBPF_MAX_REGIMES]; /* Per-regime ν */
        int student_t_learn_nu;                     /* Learn ν online */

    } RBPF_PipelineConfig;

    /* Opaque pipeline handle */
    typedef struct RBPF_Pipeline RBPF_Pipeline;

    /* Lifecycle */
    RBPF_PipelineConfig rbpf_pipeline_default_config(void);
    RBPF_Pipeline *rbpf_pipeline_create(const RBPF_PipelineConfig *config);
    void rbpf_pipeline_destroy(RBPF_Pipeline *pipe);
    void rbpf_pipeline_init(RBPF_Pipeline *pipe, rbpf_real_t initial_vol);

    /* Main API */
    void rbpf_pipeline_step(RBPF_Pipeline *pipe, rbpf_real_t ssa_return, RBPF_Signal *sig);
    void rbpf_pipeline_step_apf(RBPF_Pipeline *pipe, rbpf_real_t ssa_current,
                                rbpf_real_t ssa_next, RBPF_Signal *sig);

    /* Accessors */
    int rbpf_pipeline_get_tick(const RBPF_Pipeline *pipe);
    int rbpf_pipeline_get_regime(const RBPF_Pipeline *pipe);
    rbpf_real_t rbpf_pipeline_get_baseline_vol(const RBPF_Pipeline *pipe);
    void rbpf_pipeline_set_baseline_vol(RBPF_Pipeline *pipe, rbpf_real_t vol);
    void rbpf_pipeline_get_learned_params(const RBPF_Pipeline *pipe, int regime,
                                          rbpf_real_t *mu_vol, rbpf_real_t *sigma_vol);

    /* Runtime tuning */
    void rbpf_pipeline_set_thresholds(RBPF_Pipeline *pipe,
                                      rbpf_real_t surprise_minor, rbpf_real_t surprise_major,
                                      rbpf_real_t vol_ratio_minor, rbpf_real_t vol_ratio_major);
    void rbpf_pipeline_set_scaling(RBPF_Pipeline *pipe,
                                   rbpf_real_t scale_minor, rbpf_real_t scale_major,
                                   rbpf_real_t scale_low_conf, rbpf_real_t conf_threshold);
    void rbpf_pipeline_set_confirmation(RBPF_Pipeline *pipe, int confirm_minor, int confirm_major);

    /* Student-t runtime control */
    void rbpf_pipeline_set_student_t(RBPF_Pipeline *pipe, int enable, rbpf_real_t default_nu);
    void rbpf_pipeline_set_student_t_regime(RBPF_Pipeline *pipe, int regime, rbpf_real_t nu);

    /* Debug */
    void rbpf_pipeline_print_config(const RBPF_Pipeline *pipe);
    void rbpf_pipeline_print_signal(const RBPF_Signal *sig);

    void rbpf_ksc_set_learned_params_mode(RBPF_KSC *rbpf, int enable);

    /**
     * @brief APF step with resample index output (for Storvik integration)
     *
     * Same as rbpf_ksc_step_apf but returns the resample indices, allowing
     * external code to apply the same resampling to additional arrays.
     *
     * CRITICAL for Storvik: Without consistent resampling of per-particle
     * parameter arrays, particle states and learned params become misaligned.
     *
     * @param rbpf              RBPF context
     * @param obs_current       Current observation (raw return r_t)
     * @param obs_next          Next observation (raw return r_{t+1}) for lookahead
     * @param out               Output structure
     * @param resample_indices_out  Output: index[i] = which source particle new particle i came from
     */
    void rbpf_ksc_step_apf_indexed(
        RBPF_KSC *rbpf,
        rbpf_real_t obs_current,
        rbpf_real_t obs_next,
        RBPF_KSC_Output *out,
        int *resample_indices_out);

    /**
     * @brief Compute APF resample indices without applying
     *
     * Useful when you need to resample multiple array sets with the same indices.
     *
     * @param rbpf                  RBPF context
     * @param log_weight_combined   Combined APF weights [n_particles]
     * @param indices_out           Output: resample indices [n_particles]
     */
    void rbpf_ksc_apf_compute_resample_indices(
        RBPF_KSC *rbpf,
        const rbpf_real_t *log_weight_combined,
        int *indices_out);

    /**
     * @brief Apply pre-computed resample indices to RBPF arrays
     *
     * @param rbpf     RBPF context
     * @param indices  Resample indices from rbpf_ksc_apf_compute_resample_indices
     */
    void rbpf_ksc_apf_apply_resample_indices(
        RBPF_KSC *rbpf,
        const int *indices);

    /**
     * @brief Get last computed resample indices
     *
     * @param indices_out  Output buffer
     * @param max_n        Max indices to copy
     * @return Number of indices copied, or 0 if none available
     */
    int rbpf_ksc_apf_get_resample_indices(int *indices_out, int max_n);

#ifdef __cplusplus
}
#endif

#endif /* RBPF_KSC_H */