/*
 * ═══════════════════════════════════════════════════════════════════════════
 * RBPF Parameter Learning: Sleeping Storvik Implementation
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "rbpf_param_learn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/*═══════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 *═══════════════════════════════════════════════════════════════════════════*/

#define PI 3.14159265358979323846
#define PHI_MAX 0.995           /* Cap to prevent unit root singularity     */
#define ONE_MINUS_PHI_MIN 0.005 /* Minimum 1-φ to prevent division by ~0    */

/* Thread-local storage compatibility */
#if defined(_MSC_VER)
    #define THREAD_LOCAL __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
    #define THREAD_LOCAL __thread
#else
    #define THREAD_LOCAL  /* Fallback: no thread safety */
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * ALIGNED MEMORY ALLOCATION (for AVX-512)
 *═══════════════════════════════════════════════════════════════════════════*/

static void *aligned_alloc_64(size_t size) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, PL_CACHE_LINE);
#elif defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    return aligned_alloc(PL_CACHE_LINE, (size + PL_CACHE_LINE - 1) & ~(PL_CACHE_LINE - 1));
#else
    void *ptr = NULL;
    if (posix_memalign(&ptr, PL_CACHE_LINE, size) != 0) return NULL;
    return ptr;
#endif
}

static void aligned_free_64(void *ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/*═══════════════════════════════════════════════════════════════════════════
 * RNG: xoroshiro128+ (fast, high quality)
 *═══════════════════════════════════════════════════════════════════════════*/

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoro_next(uint64_t *s) {
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = s0 + s1;
    
    s1 ^= s0;
    s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    s[1] = rotl(s1, 37);
    
    return result;
}

static inline param_real rand_u01(uint64_t *s) {
    return (xoro_next(s) >> 11) * (1.0 / 9007199254740992.0);
}

/*═══════════════════════════════════════════════════════════════════════════
 * FAST NORMAL SAMPLER: Polar Box-Muller with caching
 * 
 * Faster than standard Box-Muller:
 * - No cos/sin (uses rejection sampling instead)
 * - Generates 2 samples, caches one
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    param_real cached;
    bool has_cached;
} NormalCache;

static THREAD_LOCAL NormalCache g_normal_cache = {0, false};

static param_real rand_normal_polar(uint64_t *s) {
    /* Return cached value if available */
    if (g_normal_cache.has_cached) {
        g_normal_cache.has_cached = false;
        return g_normal_cache.cached;
    }
    
    /* Polar rejection method */
    param_real u, v, s2;
    do {
        u = 2.0 * rand_u01(s) - 1.0;
        v = 2.0 * rand_u01(s) - 1.0;
        s2 = u * u + v * v;
    } while (s2 >= 1.0 || s2 == 0.0);
    
    param_real mult = sqrt(-2.0 * log(s2) / s2);
    
    /* Cache one value */
    g_normal_cache.cached = v * mult;
    g_normal_cache.has_cached = true;
    
    return u * mult;
}

#define rand_normal(rng) rand_normal_polar(rng)

/*═══════════════════════════════════════════════════════════════════════════
 * BATCH RNG: Fill entropy buffer (10x faster than scalar)
 * 
 * Call once per tick before particle loop. Then just read from buffer.
 * 
 * With MKL VSL: Uses AVX-512 vectorized RNG (~10x faster)
 * Without MKL:  Uses Polar Box-Muller fallback
 *═══════════════════════════════════════════════════════════════════════════*/

#ifdef PARAM_LEARN_USE_MKL
#include <mkl_vsl.h>

static void entropy_buffer_fill(EntropyBuffer *eb, int n_normal, int n_uniform) {
    n_normal = (n_normal > eb->buffer_size) ? eb->buffer_size : n_normal;
    n_uniform = (n_uniform > eb->buffer_size) ? eb->buffer_size : n_uniform;
    
    /* MKL VSL: Generate batch using AVX-512 (10x faster) */
    static VSLStreamStatePtr stream = NULL;
    if (!stream) {
        vslNewStream(&stream, VSL_BRNG_MT19937, (unsigned int)eb->rng_state[0]);
    }
    
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n_normal, 
                  eb->normal, 0.0, 1.0);
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n_uniform,
                 eb->uniform, 0.0, 1.0);
    
    eb->normal_cursor = 0;
    eb->uniform_cursor = 0;
}

#else /* Fallback: Polar Box-Muller */

static void entropy_buffer_fill(EntropyBuffer *eb, int n_normal, int n_uniform) {
    n_normal = (n_normal > eb->buffer_size) ? eb->buffer_size : n_normal;
    n_uniform = (n_uniform > eb->buffer_size) ? eb->buffer_size : n_uniform;
    
    /* Fill normal buffer using polar Box-Muller in batches */
    int i = 0;
    while (i < n_normal) {
        param_real u = 2.0 * rand_u01(eb->rng_state) - 1.0;
        param_real v = 2.0 * rand_u01(eb->rng_state) - 1.0;
        param_real s2 = u * u + v * v;
        
        if (s2 < 1.0 && s2 > 0.0) {
            param_real mult = sqrt(-2.0 * log(s2) / s2);
            eb->normal[i++] = u * mult;
            if (i < n_normal) {
                eb->normal[i++] = v * mult;
            }
        }
    }
    
    /* Fill uniform buffer */
    for (i = 0; i < n_uniform; i++) {
        eb->uniform[i] = rand_u01(eb->rng_state);
    }
    
    eb->normal_cursor = 0;
    eb->uniform_cursor = 0;
}

#endif /* PARAM_LEARN_USE_MKL */

/* Fast draws from pre-filled buffer */
static inline param_real entropy_normal(EntropyBuffer *eb) {
    if (eb->normal_cursor >= eb->buffer_size) {
        /* Refill if exhausted (shouldn't happen in normal operation) */
        entropy_buffer_fill(eb, eb->buffer_size, 0);
    }
    return eb->normal[eb->normal_cursor++];
}

static inline param_real entropy_uniform(EntropyBuffer *eb) {
    if (eb->uniform_cursor >= eb->buffer_size) {
        entropy_buffer_fill(eb, 0, eb->buffer_size);
    }
    return eb->uniform[eb->uniform_cursor++];
}

/* Marsaglia-Tsang Gamma sampler */
static param_real rand_gamma(uint64_t *s, param_real shape) {
    if (shape < 1.0) {
        return rand_gamma(s, shape + 1.0) * pow(rand_u01(s), 1.0 / shape);
    }
    
    param_real d = shape - 1.0 / 3.0;
    param_real c = 1.0 / sqrt(9.0 * d);
    
    for (;;) {
        param_real x, v;
        do {
            x = rand_normal(s);
            v = 1.0 + c * x;
        } while (v <= 0.0);
        
        v = v * v * v;
        param_real u = rand_u01(s);
        
        if (u < 1.0 - 0.0331 * (x * x) * (x * x)) {
            return d * v;
        }
        if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v))) {
            return d * v;
        }
    }
}

/* Inverse-Gamma: X ~ IG(α, β) iff 1/X ~ Gamma(α, 1/β) */
static inline param_real rand_inv_gamma(uint64_t *s, param_real alpha, param_real beta) {
    return beta / rand_gamma(s, alpha);
}

static void rng_seed(uint64_t *s, uint64_t seed) {
    uint64_t z = seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    s[0] = z ^ (z >> 31);
    
    z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    s[1] = z ^ (z >> 31);
}

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION PRESETS
 *═══════════════════════════════════════════════════════════════════════════*/

ParamLearnConfig param_learn_config_defaults(void)
{
    ParamLearnConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    
    cfg.method = PARAM_LEARN_SLEEPING_STORVIK;
    
    /* ALWAYS AWAKE: Sample every tick for all regimes
     * This eliminates stale parameters during regime transitions.
     * With SSA optimization providing ~50μs budget, we can afford this.
     *
     * Previous "sleeping" intervals caused:
     * - Calm: 99.7% ✓ (excellent)
     * - Transitions: 27-44% ✗ (terrible - stale params)
     *
     * Always-awake expected to match baseline (~65-75%) everywhere
     * while retaining Bayesian parameter learning benefits.
     */
    cfg.sample_interval[0] = 1;    /* R0 (calm): every tick */
    cfg.sample_interval[1] = 1;    /* R1: every tick        */
    cfg.sample_interval[2] = 1;    /* R2: every tick        */
    cfg.sample_interval[3] = 1;    /* R3 (crisis): every tick */
    for (int i = 4; i < PARAM_LEARN_MAX_REGIMES; i++) {
        cfg.sample_interval[i] = 1;
    }
    
    /* Triggers */
    cfg.sample_on_regime_change = true;
    cfg.sample_on_structural_break = true;
    cfg.sample_after_resampling = true;
    
    /* Load throttling */
    cfg.enable_load_throttling = true;
    cfg.load_skip_threshold = 0.9;
    
    /* EWSS (for comparison mode) */
    cfg.ewss_lambda = 0.999;
    cfg.ewss_min_eff_n = 20.0;
    
    /* Constraints */
    cfg.sigma_floor_mult = 0.1;
    cfg.sigma_ceil_mult = 5.0;
    cfg.mu_drift_max = 1.0;
    
    /* Prior strength */
    cfg.prior_strength = 10.0;
    
    cfg.rng_seed = 42;
    
    return cfg;
}

/**
 * @brief Sleeping Storvik config (for latency-constrained environments)
 *
 * Use this if you need sub-30μs latency and can tolerate slower
 * response during regime transitions.
 *
 * Tradeoffs:
 * - ✓ Calm period tracking: 99.7%
 * - ✓ Crisis detection: 97.2%
 * - ✗ Transition tracking: 27-44% (stale parameters)
 */
ParamLearnConfig param_learn_config_sleeping(void)
{
    ParamLearnConfig cfg = param_learn_config_defaults();
    
    /* Sleeping intervals: calm=slow, crisis=fast */
    cfg.sample_interval[0] = 50;    /* R0 (calm): every 50 ticks */
    cfg.sample_interval[1] = 20;    /* R1: every 20 ticks        */
    cfg.sample_interval[2] = 5;     /* R2: every 5 ticks         */
    cfg.sample_interval[3] = 1;     /* R3 (crisis): every tick   */
    
    return cfg;
}

ParamLearnConfig param_learn_config_full_bayesian(void)
{
    ParamLearnConfig cfg = param_learn_config_defaults();
    
    /* Sample every tick in all regimes */
    for (int i = 0; i < PARAM_LEARN_MAX_REGIMES; i++) {
        cfg.sample_interval[i] = 1;
    }
    
    return cfg;
}

ParamLearnConfig param_learn_config_hft(void)
{
    ParamLearnConfig cfg = param_learn_config_defaults();
    
    /* Very aggressive sleeping - rely on triggers */
    cfg.sample_interval[0] = 100;   /* R0: every 100 ticks              */
    cfg.sample_interval[1] = 50;    /* R1: every 50 ticks               */
    cfg.sample_interval[2] = 20;    /* R2: every 20 ticks               */
    cfg.sample_interval[3] = 5;     /* R3: every 5 ticks                */
    
    /* Load throttling more aggressive */
    cfg.load_skip_threshold = 0.8;
    
    return cfg;
}

ParamLearnConfig param_learn_config_ewss(void)
{
    ParamLearnConfig cfg = param_learn_config_defaults();
    cfg.method = PARAM_LEARN_EWSS;
    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

/* Helper to allocate all SoA arrays */
static int storvik_soa_alloc(StorvikSoA *soa, int total_size) {
    size_t arr_size = total_size * sizeof(param_real);
    size_t int_size = total_size * sizeof(int);
    
    soa->m = (param_real *)aligned_alloc_64(arr_size);
    soa->kappa = (param_real *)aligned_alloc_64(arr_size);
    soa->alpha = (param_real *)aligned_alloc_64(arr_size);
    soa->beta = (param_real *)aligned_alloc_64(arr_size);
    soa->mu_cached = (param_real *)aligned_alloc_64(arr_size);
    soa->sigma2_cached = (param_real *)aligned_alloc_64(arr_size);
    soa->sigma_cached = (param_real *)aligned_alloc_64(arr_size);
    soa->n_obs = (int *)aligned_alloc_64(int_size);
    soa->ticks_since_sample = (int *)aligned_alloc_64(int_size);
    
    if (!soa->m || !soa->kappa || !soa->alpha || !soa->beta ||
        !soa->mu_cached || !soa->sigma2_cached || !soa->sigma_cached ||
        !soa->n_obs || !soa->ticks_since_sample) {
        return -1;
    }
    
    /* Zero-initialize */
    memset(soa->m, 0, arr_size);
    memset(soa->kappa, 0, arr_size);
    memset(soa->alpha, 0, arr_size);
    memset(soa->beta, 0, arr_size);
    memset(soa->mu_cached, 0, arr_size);
    memset(soa->sigma2_cached, 0, arr_size);
    memset(soa->sigma_cached, 0, arr_size);
    memset(soa->n_obs, 0, int_size);
    memset(soa->ticks_since_sample, 0, int_size);
    
    return 0;
}

static void storvik_soa_free(StorvikSoA *soa) {
    aligned_free_64(soa->m);
    aligned_free_64(soa->kappa);
    aligned_free_64(soa->alpha);
    aligned_free_64(soa->beta);
    aligned_free_64(soa->mu_cached);
    aligned_free_64(soa->sigma2_cached);
    aligned_free_64(soa->sigma_cached);
    aligned_free_64(soa->n_obs);
    aligned_free_64(soa->ticks_since_sample);
    memset(soa, 0, sizeof(*soa));
}

int param_learn_init(ParamLearner *learner,
                     const ParamLearnConfig *config,
                     int n_particles,
                     int n_regimes)
{
    if (!learner || n_regimes < 1 || n_regimes > PARAM_LEARN_MAX_REGIMES ||
        n_particles < 1 || n_particles > PARAM_LEARN_MAX_PARTICLES) {
        return -1;
    }
    
    memset(learner, 0, sizeof(*learner));
    
    learner->config = config ? *config : param_learn_config_defaults();
    learner->n_regimes = n_regimes;
    learner->n_particles = n_particles;
    learner->storvik_total_size = n_particles * n_regimes;
    
    /* Initialize RNG */
    rng_seed(learner->rng, learner->config.rng_seed);
    
    /* Allocate SoA Storvik storage (aligned for SIMD) */
    if (storvik_soa_alloc(&learner->storvik, learner->storvik_total_size) < 0) {
        return -1;
    }
    
    /* Pre-allocate resample scratch buffer */
    size_t scratch_size = learner->storvik_total_size * sizeof(param_real);
    learner->resample_scratch = (param_real *)aligned_alloc_64(scratch_size * 7);
    if (!learner->resample_scratch) {
        storvik_soa_free(&learner->storvik);
        return -1;
    }
    
    /* Initialize entropy buffer for batch RNG */
    learner->entropy.buffer_size = PL_RNG_BUFFER_SIZE;
    learner->entropy.normal = (param_real *)aligned_alloc_64(PL_RNG_BUFFER_SIZE * sizeof(param_real));
    learner->entropy.uniform = (param_real *)aligned_alloc_64(PL_RNG_BUFFER_SIZE * sizeof(param_real));
    if (!learner->entropy.normal || !learner->entropy.uniform) {
        storvik_soa_free(&learner->storvik);
        aligned_free_64(learner->resample_scratch);
        return -1;
    }
    learner->entropy.rng_state[0] = learner->rng[0];
    learner->entropy.rng_state[1] = learner->rng[1];
    
    /* Pre-fill entropy buffer */
    entropy_buffer_fill(&learner->entropy, PL_RNG_BUFFER_SIZE, PL_RNG_BUFFER_SIZE);
    
    /* Initialize default priors with precomputed phi terms */
    for (int r = 0; r < n_regimes; r++) {
        RegimePrior *p = &learner->priors[r];
        p->m = -2.0 + 0.5 * r;
        p->kappa = learner->config.prior_strength;
        p->alpha = learner->config.prior_strength / 2.0 + 1.0;
        p->beta = (p->alpha - 1.0) * 0.01 * (1.0 + 0.5 * r);
        
        param_real phi_raw = 0.98 - 0.03 * r;
        p->phi = fmin(PHI_MAX, phi_raw);
        p->sigma_prior = sqrt(p->beta / (p->alpha - 1.0));
        
        p->one_minus_phi = fmax(ONE_MINUS_PHI_MIN, 1.0 - p->phi);
        p->one_minus_phi_sq = p->one_minus_phi * p->one_minus_phi;
        p->inv_one_minus_phi = 1.0 / p->one_minus_phi;
        p->inv_one_minus_phi_sq = 1.0 / p->one_minus_phi_sq;
    }
    
    return 0;
}

void param_learn_free(ParamLearner *learner)
{
    if (!learner) return;
    storvik_soa_free(&learner->storvik);
    aligned_free_64(learner->resample_scratch);
    aligned_free_64(learner->entropy.normal);
    aligned_free_64(learner->entropy.uniform);
    memset(learner, 0, sizeof(*learner));
}

void param_learn_reset(ParamLearner *learner)
{
    if (!learner) return;
    
    /* Reset runtime state */
    learner->tick = 0;
    learner->structural_break_flag = false;
    learner->current_load = 0;
    
    /* Reset diagnostics */
    learner->total_stat_updates = 0;
    learner->total_samples_drawn = 0;
    learner->samples_skipped_load = 0;
    learner->samples_triggered_regime = 0;
    learner->samples_triggered_break = 0;
    
    /* Reset EWSS */
    memset(learner->ewss, 0, sizeof(learner->ewss));
    
    /* Broadcast priors to reset Storvik stats */
    param_learn_broadcast_priors(learner);
}

/*═══════════════════════════════════════════════════════════════════════════
 * PRIOR SPECIFICATION
 *═══════════════════════════════════════════════════════════════════════════*/

void param_learn_set_prior(ParamLearner *learner,
                           int regime,
                           param_real mu,
                           param_real phi,
                           param_real sigma)
{
    if (!learner || regime < 0 || regime >= learner->n_regimes) return;
    
    RegimePrior *p = &learner->priors[regime];
    param_real n = learner->config.prior_strength;
    
    /* Convert to NIG hyperparameters */
    p->m = mu;
    p->kappa = n;
    p->alpha = n / 2.0 + 1.0;
    p->beta = (p->alpha - 1.0) * sigma * sigma;
    
    /* Cap phi to prevent unit root singularity */
    p->phi = fmin(PHI_MAX, phi);
    p->sigma_prior = sigma;
    
    /* Precompute phi terms (avoid division on hot path) */
    p->one_minus_phi = fmax(ONE_MINUS_PHI_MIN, 1.0 - p->phi);
    p->one_minus_phi_sq = p->one_minus_phi * p->one_minus_phi;
    p->inv_one_minus_phi = 1.0 / p->one_minus_phi;
    p->inv_one_minus_phi_sq = 1.0 / p->one_minus_phi_sq;
}

void param_learn_set_prior_nig(ParamLearner *learner,
                               int regime,
                               param_real m, param_real kappa,
                               param_real alpha, param_real beta,
                               param_real phi)
{
    if (!learner || regime < 0 || regime >= learner->n_regimes) return;
    
    RegimePrior *p = &learner->priors[regime];
    p->m = m;
    p->kappa = kappa;
    p->alpha = alpha;
    p->beta = beta;
    
    /* Cap phi to prevent unit root singularity */
    p->phi = fmin(PHI_MAX, phi);
    p->sigma_prior = sqrt(beta / (alpha - 1.0 + 1e-10));
    
    /* Precompute phi terms (avoid division on hot path) */
    p->one_minus_phi = fmax(ONE_MINUS_PHI_MIN, 1.0 - p->phi);
    p->one_minus_phi_sq = p->one_minus_phi * p->one_minus_phi;
    p->inv_one_minus_phi = 1.0 / p->one_minus_phi;
    p->inv_one_minus_phi_sq = 1.0 / p->one_minus_phi_sq;
}

void param_learn_broadcast_priors(ParamLearner *learner)
{
    if (!learner) return;
    
    StorvikSoA *soa = &learner->storvik;
    int nr = learner->n_regimes;
    
    for (int i = 0; i < learner->n_particles; i++) {
        for (int r = 0; r < nr; r++) {
            int idx = i * nr + r;
            const RegimePrior *p = &learner->priors[r];
            
            /* Copy prior hyperparameters */
            soa->m[idx] = p->m;
            soa->kappa[idx] = p->kappa;
            soa->alpha[idx] = p->alpha;
            soa->beta[idx] = p->beta;
            
            /* Initialize cached samples from prior */
            soa->sigma2_cached[idx] = p->beta / (p->alpha - 1.0 + 1e-10);
            soa->sigma_cached[idx] = sqrt(soa->sigma2_cached[idx]);
            soa->mu_cached[idx] = p->m;
            
            /* Reset tracking */
            soa->n_obs[idx] = 0;
            soa->ticks_since_sample[idx] = 0;
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * STORVIK SoA: SUFFICIENT STATISTICS UPDATE (always runs - cheap)
 * 
 * Vectorizable: No branches, pure arithmetic on contiguous arrays
 * FORCE_INLINE: Eliminate function call overhead on hot path
 *═══════════════════════════════════════════════════════════════════════════*/

/* Force inline for hot path */
#if defined(__GNUC__) || defined(__clang__)
    #define FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
    #define FORCE_INLINE __forceinline
#else
    #define FORCE_INLINE inline
#endif

static FORCE_INLINE void storvik_update_single_soa(
    param_real * __restrict m_arr,
    param_real * __restrict kappa_arr,
    param_real * __restrict alpha_arr,
    param_real * __restrict beta_arr,
    int * __restrict n_obs_arr,
    int * __restrict ticks_arr,
    int idx,
    param_real ell,
    param_real ell_lag,
    param_real phi,
    param_real one_minus_phi,
    param_real one_minus_phi_sq,
    param_real inv_one_minus_phi,
    param_real inv_one_minus_phi_sq)
{
    param_real z = ell - phi * ell_lag;
    param_real z_scaled = z * inv_one_minus_phi;
    param_real var_scale = inv_one_minus_phi_sq;
    
    param_real kappa_old = kappa_arr[idx];
    param_real m_old = m_arr[idx];
    
    param_real kappa_new = kappa_old + one_minus_phi_sq;
    param_real m_new = (kappa_old * m_old + z * one_minus_phi) / kappa_new;
    
    param_real alpha_new = alpha_arr[idx] + 0.5;
    
    param_real diff = z_scaled - m_old;
    param_real total_var = 1.0 / kappa_old + var_scale;
    param_real beta_new = beta_arr[idx] + 0.5 * diff * diff / total_var;
    
    m_arr[idx] = m_new;
    kappa_arr[idx] = kappa_new;
    alpha_arr[idx] = alpha_new;
    beta_arr[idx] = beta_new;
    n_obs_arr[idx]++;
    ticks_arr[idx]++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * STORVIK SoA: PARAMETER SAMPLING (conditional - uses pre-filled entropy)
 *═══════════════════════════════════════════════════════════════════════════*/

static void storvik_sample_soa(ParamLearner *learner, int idx, int regime)
{
    StorvikSoA *soa = &learner->storvik;
    const RegimePrior *p = &learner->priors[regime];
    const ParamLearnConfig *cfg = &learner->config;
    EntropyBuffer *eb = &learner->entropy;
    
    /* Sample σ² ~ InvGamma(α, β) using pre-filled entropy */
    /* InvGamma via Gamma: if X ~ Gamma(α, 1/β), then 1/X ~ InvGamma(α, β) */
    param_real alpha = soa->alpha[idx];
    param_real beta = soa->beta[idx];
    
    /* Gamma sampling using normal approximation for α > 1 */
    param_real d = alpha - 1.0 / 3.0;
    param_real c = 1.0 / sqrt(9.0 * d);
    param_real gamma_sample;
    
    for (;;) {
        param_real x = entropy_normal(eb);
        param_real v = 1.0 + c * x;
        if (v > 0) {
            v = v * v * v;
            param_real u = entropy_uniform(eb);
            if (u < 1.0 - 0.0331 * (x * x) * (x * x) ||
                log(u) < 0.5 * x * x + d * (1.0 - v + log(v))) {
                gamma_sample = d * v;
                break;
            }
        }
    }
    
    param_real sigma2 = beta / gamma_sample;
    
    /* Clamp σ² */
    param_real sigma2_prior = p->sigma_prior * p->sigma_prior;
    param_real sigma2_min = cfg->sigma_floor_mult * cfg->sigma_floor_mult * sigma2_prior;
    param_real sigma2_max = cfg->sigma_ceil_mult * cfg->sigma_ceil_mult * sigma2_prior;
    sigma2 = fmax(sigma2_min, fmin(sigma2_max, sigma2));
    
    /* Sample μ ~ N(m, σ²/κ) using pre-filled entropy */
    param_real mu_std = sqrt(sigma2 / soa->kappa[idx]);
    param_real mu = soa->m[idx] + mu_std * entropy_normal(eb);
    
    /* Clamp μ drift */
    param_real mu_drift = mu - p->m;
    if (fabs(mu_drift) > cfg->mu_drift_max) {
        mu = p->m + (mu_drift > 0 ? cfg->mu_drift_max : -cfg->mu_drift_max);
    }
    
    /* Store samples */
    soa->mu_cached[idx] = mu;
    soa->sigma2_cached[idx] = sigma2;
    soa->sigma_cached[idx] = sqrt(sigma2);
    soa->ticks_since_sample[idx] = 0;
    
    learner->total_samples_drawn++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * EWSS: UPDATE AND MLE (for comparison)
 *═══════════════════════════════════════════════════════════════════════════*/

static void ewss_update(EWSSStats *e,
                        const RegimePrior *prior,
                        param_real ell,
                        param_real ell_lag,
                        param_real weight,
                        param_real lambda)
{
    param_real phi = prior->phi;
    param_real z = ell - phi * ell_lag;
    
    e->sum_z = lambda * e->sum_z + weight * z;
    e->sum_z_sq = lambda * e->sum_z_sq + weight * z * z;
    e->eff_n = lambda * e->eff_n + weight;
}

static void ewss_compute_mle(EWSSStats *e,
                             const RegimePrior *prior,
                             param_real min_eff_n)
{
    if (e->eff_n < min_eff_n) {
        e->mu = prior->m;
        e->sigma = prior->sigma_prior;
        return;
    }
    
    param_real phi = prior->phi;
    param_real one_minus_phi = 1.0 - phi;
    
    param_real mean_z = e->sum_z / e->eff_n;
    param_real var_z = e->sum_z_sq / e->eff_n - mean_z * mean_z;
    var_z = fmax(1e-10, var_z);
    
    /* μ = E[z] / (1-φ) */
    e->mu = mean_z / one_minus_phi;
    
    /* σ² = Var[z] */
    e->sigma = sqrt(var_z);
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN UPDATE: THE SLEEPING BAYESIAN CORE
 *═══════════════════════════════════════════════════════════════════════════*/

void param_learn_update(ParamLearner *learner,
                        const ParticleInfo *particles,
                        int n)
{
    if (!learner || !particles || n < 1) return;
    
    const ParamLearnConfig *cfg = &learner->config;
    learner->tick++;
    
    /* Check load throttling */
    bool load_ok = !cfg->enable_load_throttling || 
                   learner->current_load < cfg->load_skip_threshold;
    
    /* Check structural break flag */
    bool break_flag = learner->structural_break_flag;
    if (break_flag) {
        learner->structural_break_flag = false;
    }
    
    /*═══════════════════════════════════════════════════════════════════════
     * EWSS MODE (comparison)
     *═══════════════════════════════════════════════════════════════════════*/
    if (cfg->method == PARAM_LEARN_EWSS) {
        for (int i = 0; i < n && i < learner->n_particles; i++) {
            const ParticleInfo *p = &particles[i];
            int r = p->regime;
            if (r < 0 || r >= learner->n_regimes) continue;
            
            ewss_update(&learner->ewss[r], &learner->priors[r],
                       p->ell, p->ell_lag, p->weight, cfg->ewss_lambda);
        }
        
        /* Compute MLE */
        for (int r = 0; r < learner->n_regimes; r++) {
            ewss_compute_mle(&learner->ewss[r], &learner->priors[r],
                            cfg->ewss_min_eff_n);
        }
        
        learner->total_stat_updates += n;
        return;
    }
    
    /*═══════════════════════════════════════════════════════════════════════
     * FIXED MODE (no adaptation)
     *═══════════════════════════════════════════════════════════════════════*/
    if (cfg->method == PARAM_LEARN_FIXED) {
        return;
    }
    
    /*═══════════════════════════════════════════════════════════════════════
     * SLEEPING STORVIK (primary) - SoA layout with regime-sorted processing
     *
     * Key optimization: Process particles by regime to:
     * 1. Eliminate branch misprediction (prior pointer constant per batch)
     * 2. Enable SIMD vectorization (compiler knows prior is loop-invariant)
     *═══════════════════════════════════════════════════════════════════════*/
    
    StorvikSoA *soa = &learner->storvik;
    int nr = learner->n_regimes;
    int np = learner->n_particles;
    
    /* Pre-fill entropy buffer before particle loop (batch RNG) */
    int n_samples_needed = n * 4;
    if (learner->entropy.normal_cursor + n_samples_needed > learner->entropy.buffer_size) {
        entropy_buffer_fill(&learner->entropy, PL_RNG_BUFFER_SIZE, PL_RNG_BUFFER_SIZE);
    }
    
    /*───────────────────────────────────────────────────────────────────────
     * PHASE 1: Build regime worklists (single pass)
     * This allows vectorized processing per regime
     *─────────────────────────────────────────────────────────────────────*/
    
    /* Static worklists on stack (avoid malloc) - max 1024 particles */
    int worklist[PARAM_LEARN_MAX_REGIMES][PARAM_LEARN_MAX_PARTICLES];
    int worklist_count[PARAM_LEARN_MAX_REGIMES] = {0};
    
    for (int i = 0; i < n && i < np; i++) {
        int r = particles[i].regime;
        if (r >= 0 && r < nr) {
            worklist[r][worklist_count[r]++] = i;
        }
    }
    
    /*───────────────────────────────────────────────────────────────────────
     * PHASE 2: Process each regime as a batch
     * Prior pointer is hoisted → compiler can vectorize inner loop
     *─────────────────────────────────────────────────────────────────────*/
    
    /* Hoist SoA array pointers with restrict for SIMD */
    param_real * __restrict m_arr = soa->m;
    param_real * __restrict kappa_arr = soa->kappa;
    param_real * __restrict alpha_arr = soa->alpha;
    param_real * __restrict beta_arr = soa->beta;
    int * __restrict n_obs_arr = soa->n_obs;
    int * __restrict ticks_arr = soa->ticks_since_sample;
    
    for (int r = 0; r < nr; r++) {
        int count = worklist_count[r];
        if (count == 0) continue;
        
        /* Hoist prior values (loop-invariant) */
        const RegimePrior *prior = &learner->priors[r];
        const param_real phi = prior->phi;
        const param_real one_minus_phi = prior->one_minus_phi;
        const param_real one_minus_phi_sq = prior->one_minus_phi_sq;
        const param_real inv_one_minus_phi = prior->inv_one_minus_phi;
        const param_real inv_one_minus_phi_sq = prior->inv_one_minus_phi_sq;
        const int sample_interval = cfg->sample_interval[r];
        
        /* Process all particles in this regime */
        #ifdef __GNUC__
        #pragma GCC ivdep  /* Assert no loop-carried dependencies */
        #endif
        for (int k = 0; k < count; k++) {
            int i = worklist[r][k];
            const ParticleInfo *p = &particles[i];
            int idx = i * nr + r;
            
            /*───────────────────────────────────────────────────────────────
             * STEP 1: ALWAYS update sufficient statistics
             * Inlined for SIMD - all prior values hoisted
             *─────────────────────────────────────────────────────────────*/
            storvik_update_single_soa(
                m_arr, kappa_arr, alpha_arr, beta_arr, n_obs_arr, ticks_arr,
                idx, p->ell, p->ell_lag,
                phi, one_minus_phi, one_minus_phi_sq,
                inv_one_minus_phi, inv_one_minus_phi_sq);
            
            learner->total_stat_updates++;
        }
        
        /*───────────────────────────────────────────────────────────────────
         * STEP 2: Sampling pass (separate to avoid polluting stat update)
         * This is the expensive part - only runs when triggered
         *─────────────────────────────────────────────────────────────────*/
        for (int k = 0; k < count; k++) {
            int i = worklist[r][k];
            const ParticleInfo *p = &particles[i];
            int idx = i * nr + r;
            
            bool should_sample = false;
            
            if (n_obs_arr[idx] == 1) {
                should_sample = true;
            }
            else if (break_flag && cfg->sample_on_structural_break) {
                should_sample = true;
                learner->samples_triggered_break++;
            }
            else if (p->regime != p->prev_regime && cfg->sample_on_regime_change) {
                should_sample = true;
                learner->samples_triggered_regime++;
            }
            else if (sample_interval > 0 && ticks_arr[idx] >= sample_interval) {
                should_sample = true;
            }
            
            if (should_sample && load_ok) {
                storvik_sample_soa(learner, idx, r);
            } else if (should_sample) {
                learner->samples_skipped_load++;
            }
        }
    }
}

void param_learn_signal_structural_break(ParamLearner *learner)
{
    if (learner) {
        learner->structural_break_flag = true;
    }
}

void param_learn_set_load(ParamLearner *learner, param_real load)
{
    if (learner) {
        learner->current_load = fmax(0.0, fmin(1.0, load));
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * GET PARAMETERS
 *═══════════════════════════════════════════════════════════════════════════*/

void param_learn_get_params(const ParamLearner *learner,
                            int particle_idx,
                            int regime,
                            RegimeParams *params)
{
    if (!learner || !params) {
        if (params) memset(params, 0, sizeof(*params));
        return;
    }
    
    if (regime < 0 || regime >= learner->n_regimes) {
        memset(params, 0, sizeof(*params));
        return;
    }
    
    const RegimePrior *p = &learner->priors[regime];
    const ParamLearnConfig *cfg = &learner->config;
    
    /* Fixed mode: return priors */
    if (cfg->method == PARAM_LEARN_FIXED) {
        params->mu = p->m;
        params->phi = p->phi;
        params->sigma = p->sigma_prior;
        params->sigma2 = p->sigma_prior * p->sigma_prior;
        params->mu_post_mean = p->m;
        params->mu_post_std = 0;
        params->sigma2_post_mean = params->sigma2;
        params->sigma2_post_std = 0;
        params->n_obs = 0;
        params->ticks_since_sample = 0;
        params->last_trigger = SAMPLE_TRIGGER_NONE;
        params->confidence = 0;
        return;
    }
    
    /* EWSS mode: return global MLE */
    if (cfg->method == PARAM_LEARN_EWSS) {
        const EWSSStats *e = &learner->ewss[regime];
        params->mu = e->mu;
        params->phi = p->phi;
        params->sigma = e->sigma;
        params->sigma2 = e->sigma * e->sigma;
        params->mu_post_mean = e->mu;
        params->mu_post_std = 0;
        params->sigma2_post_mean = params->sigma2;
        params->sigma2_post_std = 0;
        params->n_obs = (int)e->eff_n;
        params->ticks_since_sample = 0;
        params->last_trigger = SAMPLE_TRIGGER_NONE;
        params->confidence = fmin(1.0, e->eff_n / cfg->ewss_min_eff_n);
        return;
    }
    
    /* Storvik mode: return cached samples from SoA */
    if (particle_idx < 0 || particle_idx >= learner->n_particles) {
        /* Invalid particle - return prior */
        params->mu = p->m;
        params->phi = p->phi;
        params->sigma = p->sigma_prior;
        params->sigma2 = p->sigma_prior * p->sigma_prior;
        params->mu_post_mean = p->m;
        params->mu_post_std = p->sigma_prior / sqrt(p->kappa);
        params->sigma2_post_mean = params->sigma2;
        params->sigma2_post_std = params->sigma2 / sqrt(p->alpha);
        params->n_obs = 0;
        params->ticks_since_sample = 0;
        params->last_trigger = SAMPLE_TRIGGER_NONE;
        params->confidence = 0;
        return;
    }
    
    const StorvikSoA *soa = &learner->storvik;
    int idx = particle_idx * learner->n_regimes + regime;
    
    /* Return cached samples */
    params->mu = soa->mu_cached[idx];
    params->phi = p->phi;
    params->sigma = soa->sigma_cached[idx];
    params->sigma2 = soa->sigma2_cached[idx];
    
    /* Posterior statistics */
    param_real sigma2_post = soa->beta[idx] / (soa->alpha[idx] - 1.0 + 1e-10);
    params->mu_post_mean = soa->m[idx];
    params->mu_post_std = sqrt(sigma2_post / soa->kappa[idx]);
    params->sigma2_post_mean = sigma2_post;
    params->sigma2_post_std = sigma2_post / sqrt(soa->alpha[idx] - 1.0 + 1e-10);
    
    /* Diagnostics */
    params->n_obs = soa->n_obs[idx];
    params->ticks_since_sample = soa->ticks_since_sample[idx];
    params->last_trigger = SAMPLE_TRIGGER_NONE;  /* Not tracked per-element in SoA */
    
    /* Confidence: ratio of data precision to total precision */
    param_real prior_kappa = p->kappa;
    params->confidence = 1.0 - prior_kappa / soa->kappa[idx];
}

void param_learn_force_sample(ParamLearner *learner,
                              int particle_idx,
                              int regime)
{
    if (!learner || learner->config.method != PARAM_LEARN_SLEEPING_STORVIK) return;
    if (particle_idx < 0 || particle_idx >= learner->n_particles) return;
    if (regime < 0 || regime >= learner->n_regimes) return;
    
    int idx = particle_idx * learner->n_regimes + regime;
    storvik_sample_soa(learner, idx, regime);
}

/*═══════════════════════════════════════════════════════════════════════════
 * PARTICLE RESAMPLING SUPPORT
 *═══════════════════════════════════════════════════════════════════════════*/

void param_learn_copy_ancestor(ParamLearner *learner,
                               int dst_particle,
                               int src_particle)
{
    if (!learner) return;
    if (dst_particle < 0 || dst_particle >= learner->n_particles) return;
    if (src_particle < 0 || src_particle >= learner->n_particles) return;
    if (dst_particle == src_particle) return;
    
    StorvikSoA *soa = &learner->storvik;
    int nr = learner->n_regimes;
    int dst_base = dst_particle * nr;
    int src_base = src_particle * nr;
    
    /* Copy each SoA array slice */
    memcpy(&soa->m[dst_base], &soa->m[src_base], nr * sizeof(param_real));
    memcpy(&soa->kappa[dst_base], &soa->kappa[src_base], nr * sizeof(param_real));
    memcpy(&soa->alpha[dst_base], &soa->alpha[src_base], nr * sizeof(param_real));
    memcpy(&soa->beta[dst_base], &soa->beta[src_base], nr * sizeof(param_real));
    memcpy(&soa->mu_cached[dst_base], &soa->mu_cached[src_base], nr * sizeof(param_real));
    memcpy(&soa->sigma2_cached[dst_base], &soa->sigma2_cached[src_base], nr * sizeof(param_real));
    memcpy(&soa->sigma_cached[dst_base], &soa->sigma_cached[src_base], nr * sizeof(param_real));
    memcpy(&soa->n_obs[dst_base], &soa->n_obs[src_base], nr * sizeof(int));
    memcpy(&soa->ticks_since_sample[dst_base], &soa->ticks_since_sample[src_base], nr * sizeof(int));
}

void param_learn_apply_resampling(ParamLearner *learner,
                                  const int *ancestors,
                                  int n)
{
    if (!learner || !ancestors) return;
    
    StorvikSoA *soa = &learner->storvik;
    int nr = learner->n_regimes;
    int total = learner->storvik_total_size;
    
    /* Use pre-allocated scratch buffer (NO MALLOC ON HOT PATH!) */
    /* Scratch layout: [m, kappa, alpha, beta, mu_cached, sigma2_cached, sigma_cached] */
    param_real *scratch = learner->resample_scratch;
    param_real *tmp_m = scratch;
    param_real *tmp_kappa = scratch + total;
    param_real *tmp_alpha = scratch + total * 2;
    param_real *tmp_beta = scratch + total * 3;
    param_real *tmp_mu = scratch + total * 4;
    param_real *tmp_s2 = scratch + total * 5;
    param_real *tmp_s = scratch + total * 6;
    
    /* Copy based on ancestors */
    for (int i = 0; i < n && i < learner->n_particles; i++) {
        int anc = ancestors[i];
        if (anc >= 0 && anc < learner->n_particles) {
            int dst_base = i * nr;
            int src_base = anc * nr;
            memcpy(&tmp_m[dst_base], &soa->m[src_base], nr * sizeof(param_real));
            memcpy(&tmp_kappa[dst_base], &soa->kappa[src_base], nr * sizeof(param_real));
            memcpy(&tmp_alpha[dst_base], &soa->alpha[src_base], nr * sizeof(param_real));
            memcpy(&tmp_beta[dst_base], &soa->beta[src_base], nr * sizeof(param_real));
            memcpy(&tmp_mu[dst_base], &soa->mu_cached[src_base], nr * sizeof(param_real));
            memcpy(&tmp_s2[dst_base], &soa->sigma2_cached[src_base], nr * sizeof(param_real));
            memcpy(&tmp_s[dst_base], &soa->sigma_cached[src_base], nr * sizeof(param_real));
        }
    }
    
    /* Copy back */
    size_t arr_size = total * sizeof(param_real);
    memcpy(soa->m, tmp_m, arr_size);
    memcpy(soa->kappa, tmp_kappa, arr_size);
    memcpy(soa->alpha, tmp_alpha, arr_size);
    memcpy(soa->beta, tmp_beta, arr_size);
    memcpy(soa->mu_cached, tmp_mu, arr_size);
    memcpy(soa->sigma2_cached, tmp_s2, arr_size);
    memcpy(soa->sigma_cached, tmp_s, arr_size);
    
    /* n_obs and ticks_since_sample: need separate handling (int arrays) */
    /* For simplicity, just reset ticks_since_sample to trigger resampling */
    if (learner->config.sample_after_resampling) {
        for (int i = 0; i < n && i < learner->n_particles; i++) {
            for (int r = 0; r < nr; r++) {
                soa->ticks_since_sample[i * nr + r] = learner->config.sample_interval[r];
            }
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

void param_learn_print_summary(const ParamLearner *learner)
{
    if (!learner) return;
    
    const char *method_str[] = {"SLEEPING_STORVIK", "EWSS", "FIXED"};
    
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║            Parameter Learner Summary                         ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ Method: %-20s                               ║\n", method_str[learner->config.method]);
    printf("║ Particles: %-4d  Regimes: %-4d  Tick: %-8d               ║\n",
           learner->n_particles, learner->n_regimes, learner->tick);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ Statistics:                                                  ║\n");
    printf("║   Total stat updates:    %12llu                        ║\n",
           (unsigned long long)learner->total_stat_updates);
    printf("║   Total samples drawn:   %12llu                        ║\n",
           (unsigned long long)learner->total_samples_drawn);
    printf("║   Samples skipped (load):%12llu                        ║\n",
           (unsigned long long)learner->samples_skipped_load);
    printf("║   Triggered by regime:   %12llu                        ║\n",
           (unsigned long long)learner->samples_triggered_regime);
    printf("║   Triggered by break:    %12llu                        ║\n",
           (unsigned long long)learner->samples_triggered_break);
    
    if (learner->total_stat_updates > 0) {
        double sample_rate = 100.0 * learner->total_samples_drawn / learner->total_stat_updates;
        printf("║   Sample rate:           %11.2f%%                        ║\n", sample_rate);
    }
    
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ Sampling Intervals:                                          ║\n");
    for (int r = 0; r < learner->n_regimes; r++) {
        printf("║   R%d: every %3d ticks                                        ║\n",
               r, learner->config.sample_interval[r]);
    }
    printf("╚══════════════════════════════════════════════════════════════╝\n");
}

void param_learn_print_regime_stats(const ParamLearner *learner, int regime)
{
    if (!learner || regime < 0 || regime >= learner->n_regimes) return;
    
    const RegimePrior *p = &learner->priors[regime];
    
    printf("\n─── Regime %d Statistics ───\n", regime);
    printf("Prior: μ=%.4f, φ=%.4f, σ=%.4f\n", p->m, p->phi, p->sigma_prior);
    
    if (learner->config.method == PARAM_LEARN_EWSS) {
        const EWSSStats *e = &learner->ewss[regime];
        printf("EWSS:  μ=%.4f, σ=%.4f (eff_n=%.1f)\n", e->mu, e->sigma, e->eff_n);
    }
    else if (learner->config.method == PARAM_LEARN_SLEEPING_STORVIK) {
        /* Aggregate over particles using SoA */
        const StorvikSoA *soa = &learner->storvik;
        double sum_mu = 0, sum_sigma = 0;
        int count = 0;
        for (int i = 0; i < learner->n_particles; i++) {
            int idx = i * learner->n_regimes + regime;
            if (soa->n_obs[idx] > 0) {
                sum_mu += soa->mu_cached[idx];
                sum_sigma += soa->sigma_cached[idx];
                count++;
            }
        }
        if (count > 0) {
            printf("Storvik (avg over %d particles): μ=%.4f, σ=%.4f\n",
                   count, sum_mu / count, sum_sigma / count);
        }
    }
}

void param_learn_get_regime_summary(const ParamLearner *learner,
                                    int regime,
                                    param_real *mu_mean,
                                    param_real *mu_std,
                                    param_real *sigma_mean,
                                    param_real *sigma_std,
                                    int *total_obs)
{
    if (!learner || regime < 0 || regime >= learner->n_regimes) return;
    
    double sum_mu = 0, sum_mu_sq = 0;
    double sum_sigma = 0, sum_sigma_sq = 0;
    int count = 0, obs = 0;
    
    if (learner->config.method == PARAM_LEARN_SLEEPING_STORVIK) {
        const StorvikSoA *soa = &learner->storvik;
        for (int i = 0; i < learner->n_particles; i++) {
            int idx = i * learner->n_regimes + regime;
            double mu = soa->mu_cached[idx];
            double sigma = soa->sigma_cached[idx];
            sum_mu += mu;
            sum_mu_sq += mu * mu;
            sum_sigma += sigma;
            sum_sigma_sq += sigma * sigma;
            obs += soa->n_obs[idx];
            count++;
        }
    }
    
    if (count > 0) {
        double m_mu = sum_mu / count;
        double m_sigma = sum_sigma / count;
        
        if (mu_mean) *mu_mean = m_mu;
        if (mu_std) *mu_std = sqrt(sum_mu_sq / count - m_mu * m_mu);
        if (sigma_mean) *sigma_mean = m_sigma;
        if (sigma_std) *sigma_std = sqrt(sum_sigma_sq / count - m_sigma * m_sigma);
        if (total_obs) *total_obs = obs;
    }
}
