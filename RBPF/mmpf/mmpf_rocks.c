/**
 * @file mmpf_rocks.c
 * @brief IMM-MMPF-ROCKS Implementation
 *
 * See mmpf_rocks.h for architecture documentation.
 */

#include "mmpf_rocks.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* SIMD intrinsics for double→float conversion */
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

/* Compiler hints */
#if defined(__GNUC__) || defined(__clang__)
#define MMPF_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define MMPF_RESTRICT __restrict
#else
#define MMPF_RESTRICT
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * INTERNAL HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

/* Aligned allocation helper */
static void *aligned_alloc_impl(size_t size, size_t alignment)
{
#ifdef _MSC_VER
    return _aligned_malloc(size, alignment);
#else
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0)
    {
        return NULL;
    }
    return ptr;
#endif
}

static void aligned_free_impl(void *ptr)
{
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/* Normalize weights (in-place) */
static void normalize_weights(rbpf_real_t *weights, int n)
{
    rbpf_real_t sum = RBPF_REAL(0.0);
    for (int i = 0; i < n; i++)
    {
        sum += weights[i];
    }
    if (sum > RBPF_EPS)
    {
        rbpf_real_t inv_sum = RBPF_REAL(1.0) / sum;
        for (int i = 0; i < n; i++)
        {
            weights[i] *= inv_sum;
        }
    }
    else
    {
        /* Uniform fallback */
        rbpf_real_t uniform = RBPF_REAL(1.0) / n;
        for (int i = 0; i < n; i++)
        {
            weights[i] = uniform;
        }
    }
}

/* Log-sum-exp for numerical stability */
static rbpf_real_t log_sum_exp(const rbpf_real_t *log_w, int n)
{
    rbpf_real_t max_log = log_w[0];
    for (int i = 1; i < n; i++)
    {
        if (log_w[i] > max_log)
            max_log = log_w[i];
    }

    rbpf_real_t sum = RBPF_REAL(0.0);
    for (int i = 0; i < n; i++)
    {
        sum += rbpf_exp(log_w[i] - max_log);
    }

    return max_log + rbpf_log(sum);
}

/* Normalize log-weights to linear weights */
static void log_weights_to_linear(const rbpf_real_t *log_w, rbpf_real_t *w, int n)
{
    rbpf_real_t lse = log_sum_exp(log_w, n);
    for (int i = 0; i < n; i++)
    {
        w[i] = rbpf_exp(log_w[i] - lse);
    }
}

/* Argmax */
static int argmax(const rbpf_real_t *arr, int n)
{
    int idx = 0;
    rbpf_real_t max_val = arr[0];
    for (int i = 1; i < n; i++)
    {
        if (arr[i] > max_val)
        {
            max_val = arr[i];
            idx = i;
        }
    }
    return idx;
}

/*═══════════════════════════════════════════════════════════════════════════
 * PARTICLE BUFFER IMPLEMENTATION
 *═══════════════════════════════════════════════════════════════════════════*/

MMPF_ParticleBuffer *mmpf_buffer_create(int n_particles, int n_storvik_regimes)
{
    MMPF_ParticleBuffer *buf = (MMPF_ParticleBuffer *)malloc(sizeof(MMPF_ParticleBuffer));
    if (!buf)
        return NULL;

    memset(buf, 0, sizeof(MMPF_ParticleBuffer));
    buf->n_particles = n_particles;
    buf->n_storvik_regimes = n_storvik_regimes;

    /* Calculate sizes */
    size_t n = (size_t)n_particles;
    size_t nr = (size_t)n_storvik_regimes;

    size_t rbpf_float_size = n * sizeof(rbpf_real_t);
    size_t rbpf_int_size = n * sizeof(int);
    size_t storvik_size = n * nr * sizeof(param_real);

    /* Total size with alignment padding */
    size_t total = 0;
    total += (rbpf_float_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1); /* mu */
    total += (rbpf_float_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1); /* var */
    total += (rbpf_int_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);   /* ksc_regime */
    total += (rbpf_float_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1); /* log_weight */
    total += (storvik_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);    /* storvik_m */
    total += (storvik_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);    /* storvik_kappa */
    total += (storvik_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);    /* storvik_alpha */
    total += (storvik_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);    /* storvik_beta */
    total += (storvik_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);    /* storvik_mu */
    total += (storvik_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);    /* storvik_sigma */

    buf->_memory = aligned_alloc_impl(total, MMPF_ALIGN);
    if (!buf->_memory)
    {
        free(buf);
        return NULL;
    }
    buf->_memory_size = total;
    memset(buf->_memory, 0, total);

    /* Assign pointers */
    char *ptr = (char *)buf->_memory;

    buf->mu = (rbpf_real_t *)ptr;
    ptr += (rbpf_float_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);

    buf->var = (rbpf_real_t *)ptr;
    ptr += (rbpf_float_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);

    buf->ksc_regime = (int *)ptr;
    ptr += (rbpf_int_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);

    buf->log_weight = (rbpf_real_t *)ptr;
    ptr += (rbpf_float_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);

    buf->storvik_m = (param_real *)ptr;
    ptr += (storvik_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);

    buf->storvik_kappa = (param_real *)ptr;
    ptr += (storvik_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);

    buf->storvik_alpha = (param_real *)ptr;
    ptr += (storvik_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);

    buf->storvik_beta = (param_real *)ptr;
    ptr += (storvik_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);

    buf->storvik_mu = (param_real *)ptr;
    ptr += (storvik_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);

    buf->storvik_sigma = (param_real *)ptr;

    return buf;
}

void mmpf_buffer_destroy(MMPF_ParticleBuffer *buf)
{
    if (buf)
    {
        if (buf->_memory)
            aligned_free_impl(buf->_memory);
        free(buf);
    }
}

void mmpf_buffer_export(MMPF_ParticleBuffer *buf,
                        const RBPF_KSC *rbpf,
                        const ParamLearner *learner)
{
    const int n = buf->n_particles;
    const int nr = buf->n_storvik_regimes;

    /* Export RBPF state */
    memcpy(buf->mu, rbpf->mu, n * sizeof(rbpf_real_t));
    memcpy(buf->var, rbpf->var, n * sizeof(rbpf_real_t));
    memcpy(buf->ksc_regime, rbpf->regime, n * sizeof(int));
    memcpy(buf->log_weight, rbpf->log_weight, n * sizeof(rbpf_real_t));

    /* Export Storvik stats */
    if (learner)
    {
        const StorvikSoA *soa = param_learn_get_active_soa_const(learner);
        const size_t storvik_size = (size_t)n * nr * sizeof(param_real);

        memcpy(buf->storvik_m, soa->m, storvik_size);
        memcpy(buf->storvik_kappa, soa->kappa, storvik_size);
        memcpy(buf->storvik_alpha, soa->alpha, storvik_size);
        memcpy(buf->storvik_beta, soa->beta, storvik_size);
        memcpy(buf->storvik_mu, soa->mu_cached, storvik_size);
        memcpy(buf->storvik_sigma, soa->sigma_cached, storvik_size);
    }
}

void mmpf_buffer_import(const MMPF_ParticleBuffer *buf,
                        RBPF_KSC *rbpf,
                        ParamLearner *learner)
{
    const int n = buf->n_particles;
    const int nr = buf->n_storvik_regimes;

    /* Import RBPF state */
    memcpy(rbpf->mu, buf->mu, n * sizeof(rbpf_real_t));
    memcpy(rbpf->var, buf->var, n * sizeof(rbpf_real_t));
    memcpy(rbpf->regime, buf->ksc_regime, n * sizeof(int));
    memcpy(rbpf->log_weight, buf->log_weight, n * sizeof(rbpf_real_t));

    /* Import Storvik stats */
    if (learner)
    {
        StorvikSoA *soa = param_learn_get_active_soa(learner);
        const size_t storvik_size = (size_t)n * nr * sizeof(param_real);

        memcpy(soa->m, buf->storvik_m, storvik_size);
        memcpy(soa->kappa, buf->storvik_kappa, storvik_size);
        memcpy(soa->alpha, buf->storvik_alpha, storvik_size);
        memcpy(soa->beta, buf->storvik_beta, storvik_size);
        memcpy(soa->mu_cached, buf->storvik_mu, storvik_size);
        memcpy(soa->sigma_cached, buf->storvik_sigma, storvik_size);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * PARAMETER SYNC: Bridge Storvik → RBPF
 *
 * CRITICAL: This function bridges the gap between:
 *   1. ParamLearner (Storvik) → holds learned mu_cached, sigma_cached
 *   2. RBPF_KSC → uses particle_mu_vol, particle_sigma_vol in predict step
 *
 * Without this sync, rbpf_ksc_predict() uses stale/uninitialized parameters!
 *
 * TYPE MISMATCH WARNING:
 *   param_real (Storvik) = double  (needs precision for sufficient stats)
 *   rbpf_real_t (RBPF)   = float   (for SIMD efficiency)
 *
 * Cannot use memcpy! Must convert double→float properly.
 * Uses SIMD for performance (matches glue layer's convert_double_to_float_aligned).
 *
 * Call this AFTER mmpf_buffer_import() and BEFORE rbpf_ksc_step().
 *═══════════════════════════════════════════════════════════════════════════*/

/* SIMD double→float conversion (same as glue layer) */
static void mmpf_convert_double_to_float(
    const param_real *MMPF_RESTRICT src,
    rbpf_real_t *MMPF_RESTRICT dst,
    int n)
{
#if defined(__AVX512F__) && !defined(_MSC_VER)
    /* AVX-512: 8 doubles → 8 floats per iteration */
    int i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m512d vd = _mm512_loadu_pd(src + i);
        __m256 vf = _mm512_cvtpd_ps(vd);
        _mm256_storeu_ps(dst + i, vf);
    }
    for (; i < n; i++)
    {
        dst[i] = (rbpf_real_t)src[i];
    }

#elif defined(__AVX2__)
    /* AVX2: 4 doubles → 4 floats per iteration */
    int i = 0;
    for (; i + 4 <= n; i += 4)
    {
        __m256d vd = _mm256_loadu_pd(src + i);
        __m128 vf = _mm256_cvtpd_ps(vd);
        _mm_storeu_ps(dst + i, vf);
    }
    for (; i < n; i++)
    {
        dst[i] = (rbpf_real_t)src[i];
    }

#else
    /* Scalar fallback */
    for (int i = 0; i < n; i++)
    {
        dst[i] = (rbpf_real_t)src[i];
    }
#endif
}

static void mmpf_sync_parameters(RBPF_KSC *rbpf, const ParamLearner *learner)
{
    if (!rbpf || !learner)
        return;

    const StorvikSoA *soa = param_learn_get_active_soa_const(learner);
    const int n = rbpf->n_particles;
    const int n_regimes = learner->n_regimes;

    /* Verify dimensions match (defensive) */
    if (n_regimes != rbpf->n_regimes)
    {
        /* Dimension mismatch - this shouldn't happen if configured correctly */
        /* Fall back to using global params */
        return;
    }

    const int total = n * n_regimes;

    /* Convert double (Storvik) → float (RBPF) using SIMD
     *
     * Layout: both are [n_particles * n_regimes] particle-major
     * StorvikSoA: mu_cached[particle * n_regimes + regime]  (param_real = double)
     * RBPF_KSC:   particle_mu_vol[particle * n_regimes + regime]  (rbpf_real_t = float)
     *
     * CANNOT use memcpy - types differ!
     */
    mmpf_convert_double_to_float(soa->mu_cached, rbpf->particle_mu_vol, total);
    mmpf_convert_double_to_float(soa->sigma_cached, rbpf->particle_sigma_vol, total);

    /* Store fence: ensure writes commit before next tick's predict() reads them */
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE2__)
    _mm_sfence();
#endif

    /* CRITICAL: Tell RBPF to USE per-particle parameters instead of global
     *
     * If use_learned_params == 0, rbpf_ksc_predict() uses static global params.
     * We want it to use the per-particle learned values from Storvik.
     */
    rbpf->use_learned_params = 1;
}

void mmpf_buffer_copy_particle(MMPF_ParticleBuffer *dst, int dst_idx,
                               const MMPF_ParticleBuffer *src, int src_idx)
{
    const int nr = src->n_storvik_regimes;

    /* RBPF state */
    dst->mu[dst_idx] = src->mu[src_idx];
    dst->var[dst_idx] = src->var[src_idx];
    dst->ksc_regime[dst_idx] = src->ksc_regime[src_idx];
    dst->log_weight[dst_idx] = src->log_weight[src_idx];

    /* Storvik stats */
    const int src_off = src_idx * nr;
    const int dst_off = dst_idx * nr;

    memcpy(&dst->storvik_m[dst_off], &src->storvik_m[src_off], nr * sizeof(param_real));
    memcpy(&dst->storvik_kappa[dst_off], &src->storvik_kappa[src_off], nr * sizeof(param_real));
    memcpy(&dst->storvik_alpha[dst_off], &src->storvik_alpha[src_off], nr * sizeof(param_real));
    memcpy(&dst->storvik_beta[dst_off], &src->storvik_beta[src_off], nr * sizeof(param_real));
    memcpy(&dst->storvik_mu[dst_off], &src->storvik_mu[src_off], nr * sizeof(param_real));
    memcpy(&dst->storvik_sigma[dst_off], &src->storvik_sigma[src_off], nr * sizeof(param_real));
}

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION DEFAULTS
 *═══════════════════════════════════════════════════════════════════════════*/

MMPF_Config mmpf_config_defaults(void)
{
    MMPF_Config cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.n_particles = 512;
    cfg.n_ksc_regimes = 4;
    cfg.n_storvik_regimes = 4;

    /* Calm: tight mean-reversion */
    cfg.hypotheses[MMPF_CALM].mu_vol = RBPF_REAL(-1.0);
    cfg.hypotheses[MMPF_CALM].phi = RBPF_REAL(0.98);
    cfg.hypotheses[MMPF_CALM].sigma_eta = RBPF_REAL(0.10);

    /* Trend: moderate */
    cfg.hypotheses[MMPF_TREND].mu_vol = RBPF_REAL(-0.5);
    cfg.hypotheses[MMPF_TREND].phi = RBPF_REAL(0.95);
    cfg.hypotheses[MMPF_TREND].sigma_eta = RBPF_REAL(0.20);

    /* Crisis: explosive */
    cfg.hypotheses[MMPF_CRISIS].mu_vol = RBPF_REAL(0.0);
    cfg.hypotheses[MMPF_CRISIS].phi = RBPF_REAL(0.80);
    cfg.hypotheses[MMPF_CRISIS].sigma_eta = RBPF_REAL(0.50);

    /* IMM settings */
    cfg.base_stickiness = RBPF_REAL(0.98);
    cfg.min_stickiness = RBPF_REAL(0.85);
    cfg.crisis_exit_boost = RBPF_REAL(0.92);
    cfg.min_mixing_prob = RBPF_REAL(0.01); /* 1% minimum transition probability */
    cfg.enable_adaptive_stickiness = 1;
    cfg.enable_storvik_sync = 1; /* Enable by default; disable for hypothesis discrimination tests */

    /* Zero return handling (HFT critical)
     * Policy: 0=skip update, 1=use floor, 2=censored interval (not implemented)
     * Default: use floor at log(min_tick²) ≈ -18.0 for typical HFT instruments */
    cfg.zero_return_policy = 1;               /* Use floor */
    cfg.min_log_return_sq = RBPF_REAL(-18.0); /* ~exp(-9) ≈ 0.0001 = 1bp */

    /* Initial weights: slight bias toward calm */
    cfg.initial_weights[MMPF_CALM] = RBPF_REAL(0.6);
    cfg.initial_weights[MMPF_TREND] = RBPF_REAL(0.3);
    cfg.initial_weights[MMPF_CRISIS] = RBPF_REAL(0.1);

    /* Robust OCSN defaults */
    cfg.robust_ocsn.enabled = 1;
    for (int r = 0; r < PARAM_LEARN_MAX_REGIMES; r++)
    {
        cfg.robust_ocsn.regime[r].prob = RBPF_REAL(0.01);
        cfg.robust_ocsn.regime[r].variance = RBPF_OUTLIER_VAR_DEFAULT;
    }

    /* Storvik config */
    cfg.storvik_config = param_learn_config_defaults();

    /* RNG */
    cfg.rng_seed = 42;

    return cfg;
}

MMPF_Config mmpf_config_hft(void)
{
    MMPF_Config cfg = mmpf_config_defaults();

    cfg.n_particles = 256; /* Fewer particles */
    cfg.storvik_config = param_learn_config_hft();

    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════
 * IMM MIXING INTERNALS
 *═══════════════════════════════════════════════════════════════════════════*/

/* Update transition matrix based on OCSN outlier fraction */
static void update_transition_matrix(MMPF_ROCKS *mmpf)
{
    if (!mmpf->config.enable_adaptive_stickiness)
    {
        return;
    }

    /* Compute stickiness from outlier fraction */
    rbpf_real_t base = mmpf->config.base_stickiness;
    rbpf_real_t min_s = mmpf->config.min_stickiness;
    rbpf_real_t outlier = mmpf->outlier_fraction;
    rbpf_real_t min_mix = mmpf->config.min_mixing_prob;

    /* Higher outlier fraction → lower stickiness */
    rbpf_real_t stickiness = base - (base - min_s) * outlier;
    mmpf->current_stickiness = stickiness;

    /* Build transition matrix with minimum mixing probability
     * This prevents regime lock-in by ensuring each model always has
     * at least min_mixing_prob chance of receiving particles */
    for (int i = 0; i < MMPF_N_MODELS; i++)
    {
        rbpf_real_t stay = stickiness;

        /* Crisis exits faster */
        if (i == MMPF_CRISIS)
        {
            stay *= mmpf->config.crisis_exit_boost;
        }

        /* Compute base transition probability */
        rbpf_real_t leave = (RBPF_REAL(1.0) - stay) / (MMPF_N_MODELS - 1);

        /* Apply minimum mixing probability floor */
        if (leave < min_mix)
        {
            leave = min_mix;
            /* Adjust stay probability to maintain normalization */
            stay = RBPF_REAL(1.0) - leave * (MMPF_N_MODELS - 1);
        }

        for (int j = 0; j < MMPF_N_MODELS; j++)
        {
            mmpf->transition[i][j] = (i == j) ? stay : leave;
        }
    }
}

/* Compute IMM mixing weights: μ[target][source] */
static void compute_mixing_weights(MMPF_ROCKS *mmpf)
{
    /* μ[i][j] = P(model j → model i) = π[j][i] × w[j] / Σ_k(π[k][i] × w[k]) */

    for (int target = 0; target < MMPF_N_MODELS; target++)
    {
        rbpf_real_t denom = RBPF_REAL(0.0);

        for (int source = 0; source < MMPF_N_MODELS; source++)
        {
            /* π[source][target] × w[source] */
            rbpf_real_t val = mmpf->transition[source][target] * mmpf->weights[source];
            mmpf->mixing_weights[target][source] = val;
            denom += val;
        }

        /* Normalize */
        if (denom > RBPF_EPS)
        {
            for (int source = 0; source < MMPF_N_MODELS; source++)
            {
                mmpf->mixing_weights[target][source] /= denom;
            }
        }
        else
        {
            /* Uniform fallback */
            for (int source = 0; source < MMPF_N_MODELS; source++)
            {
                mmpf->mixing_weights[target][source] = RBPF_REAL(1.0) / MMPF_N_MODELS;
            }
        }
    }
}

/* Compute stratified mixing counts */
static void compute_mixing_counts(MMPF_ROCKS *mmpf)
{
    const int n = mmpf->n_particles;

    for (int target = 0; target < MMPF_N_MODELS; target++)
    {
        int total = 0;
        int max_source = 0;
        rbpf_real_t max_weight = mmpf->mixing_weights[target][0];

        for (int source = 0; source < MMPF_N_MODELS; source++)
        {
            rbpf_real_t w = mmpf->mixing_weights[target][source];
            int count = (int)(w * n);
            mmpf->mix_counts[target][source] = count;
            total += count;

            if (w > max_weight)
            {
                max_weight = w;
                max_source = source;
            }
        }

        /* Assign remainder to largest source */
        mmpf->mix_counts[target][max_source] += (n - total);
    }
}

/* Stratified resample from source buffer into particle indices */
static void stratified_resample_from_buffer(
    const MMPF_ParticleBuffer *src,
    int *indices_out, /* Output: source particle indices */
    int count,        /* How many to draw */
    rbpf_pcg32_t *rng)
{
    const int n_src = src->n_particles;

    /* Compute cumulative weights from log_weight */
    rbpf_real_t max_log = src->log_weight[0];
    for (int i = 1; i < n_src; i++)
    {
        if (src->log_weight[i] > max_log)
            max_log = src->log_weight[i];
    }

    rbpf_real_t sum = RBPF_REAL(0.0);
    rbpf_real_t cumsum[1024]; /* Stack alloc, assume n_src <= 1024 */

    for (int i = 0; i < n_src; i++)
    {
        sum += rbpf_exp(src->log_weight[i] - max_log);
        cumsum[i] = sum;
    }

    /* Stratified resampling */
    rbpf_real_t step = sum / count;
    rbpf_real_t u = rbpf_pcg32_uniform(rng) * step;
    int src_idx = 0;

    for (int i = 0; i < count; i++)
    {
        while (src_idx < n_src - 1 && cumsum[src_idx] < u)
        {
            src_idx++;
        }
        indices_out[i] = src_idx;
        u += step;
    }
}

/* Perform IMM mixing step */
static void imm_mixing_step(MMPF_ROCKS *mmpf)
{
    const int n = mmpf->n_particles;
    const int nr = mmpf->config.n_storvik_regimes;

    /* 1. Export particles from all models to buffers */
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf_buffer_export(mmpf->buffer[k], mmpf->rbpf[k], mmpf->learner[k]);
    }

    /* 2. For each target model, draw particles from combined pool */
    int indices[1024]; /* Temp storage for resampled indices */

    for (int target = 0; target < MMPF_N_MODELS; target++)
    {
        int p_idx = 0;

        for (int source = 0; source < MMPF_N_MODELS; source++)
        {
            int count = mmpf->mix_counts[target][source];
            if (count <= 0)
                continue;

            /* Resample 'count' particles from source */
            stratified_resample_from_buffer(
                mmpf->buffer[source],
                indices,
                count,
                &mmpf->rng);

            /* Copy resampled particles to mixed buffer */
            for (int i = 0; i < count; i++)
            {
                mmpf_buffer_copy_particle(
                    mmpf->mixed_buffer[target], p_idx,
                    mmpf->buffer[source], indices[i]);

                /* Reset weight to uniform (will be reweighted in update) */
                mmpf->mixed_buffer[target]->log_weight[p_idx] = RBPF_REAL(0.0);

                p_idx++;
            }
        }
    }

    /* 3. Import mixed particles back into models */
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf_buffer_import(mmpf->mixed_buffer[k], mmpf->rbpf[k], mmpf->learner[k]);
    }

    mmpf->imm_mix_count++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

MMPF_ROCKS *mmpf_create(const MMPF_Config *config)
{
    MMPF_Config cfg = config ? *config : mmpf_config_defaults();

    MMPF_ROCKS *mmpf = (MMPF_ROCKS *)malloc(sizeof(MMPF_ROCKS));
    if (!mmpf)
        return NULL;

    memset(mmpf, 0, sizeof(MMPF_ROCKS));
    mmpf->config = cfg;
    mmpf->n_particles = cfg.n_particles;

    /* Initialize RNG */
    rbpf_pcg32_seed(&mmpf->rng, cfg.rng_seed, 1);

    /* Create 3 RBPF instances */
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->rbpf[k] = rbpf_ksc_create(cfg.n_particles, cfg.n_ksc_regimes);
        if (!mmpf->rbpf[k])
        {
            mmpf_destroy(mmpf);
            return NULL;
        }

        /* Configure hypothesis parameters */
        MMPF_HypothesisParams *hp = &cfg.hypotheses[k];
        for (int r = 0; r < cfg.n_ksc_regimes; r++)
        {
            /* Use same parameters for all KSC regimes within a hypothesis */
            /* The KSC regimes handle the mixture approximation, not volatility regimes */
            rbpf_ksc_set_regime_params(mmpf->rbpf[k], r,
                                       RBPF_REAL(1.0) - hp->phi, /* theta = 1 - phi */
                                       hp->mu_vol,
                                       hp->sigma_eta);
        }

        /* Initialize particle state to hypothesis mu_vol */
        rbpf_ksc_init(mmpf->rbpf[k], hp->mu_vol, RBPF_REAL(0.5));
    }

    /* Create 3 Storvik learners */
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->learner[k] = (ParamLearner *)malloc(sizeof(ParamLearner));
        if (!mmpf->learner[k])
        {
            mmpf_destroy(mmpf);
            return NULL;
        }

        if (param_learn_init(mmpf->learner[k], &cfg.storvik_config,
                             cfg.n_particles, cfg.n_storvik_regimes) != 0)
        {
            mmpf_destroy(mmpf);
            return NULL;
        }

        /* Set priors based on hypothesis */
        MMPF_HypothesisParams *hp = &cfg.hypotheses[k];
        for (int r = 0; r < cfg.n_storvik_regimes; r++)
        {
            param_learn_set_prior(mmpf->learner[k], r,
                                  (param_real)hp->mu_vol,
                                  (param_real)hp->phi,
                                  (param_real)hp->sigma_eta);
        }
        param_learn_broadcast_priors(mmpf->learner[k]);

        /* CRITICAL: Sync initial params from Storvik → RBPF
         * Only if learning sync is enabled. When disabled (for unit tests),
         * RBPF uses fixed global hypothesis params for discrimination. */
        if (cfg.enable_storvik_sync)
        {
            mmpf_sync_parameters(mmpf->rbpf[k], mmpf->learner[k]);
        }
    }

    /* Create particle buffers for IMM mixing */
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->buffer[k] = mmpf_buffer_create(cfg.n_particles, cfg.n_storvik_regimes);
        mmpf->mixed_buffer[k] = mmpf_buffer_create(cfg.n_particles, cfg.n_storvik_regimes);

        if (!mmpf->buffer[k] || !mmpf->mixed_buffer[k])
        {
            mmpf_destroy(mmpf);
            return NULL;
        }
    }

    /* Initialize model weights */
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->weights[k] = cfg.initial_weights[k];
        mmpf->log_weights[k] = rbpf_log(cfg.initial_weights[k]);
    }
    normalize_weights(mmpf->weights, MMPF_N_MODELS);

    /* Initialize transition matrix */
    mmpf->current_stickiness = cfg.base_stickiness;
    mmpf->outlier_fraction = RBPF_REAL(0.0);
    update_transition_matrix(mmpf);

    /* Copy robust OCSN config */
    mmpf->robust_ocsn = cfg.robust_ocsn;

    /* Initialize regime tracking */
    mmpf->dominant = MMPF_CALM;
    mmpf->prev_dominant = MMPF_CALM;
    mmpf->ticks_in_regime = 0;

    /* Initialize cached outputs from RBPF initial state */
    mmpf->weighted_vol = RBPF_REAL(0.0);
    mmpf->weighted_log_vol = RBPF_REAL(0.0);
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        /* Compute initial vol estimate from particle mean */
        RBPF_KSC *rbpf = mmpf->rbpf[k];
        rbpf_real_t sum_log_vol = RBPF_REAL(0.0);
        for (int i = 0; i < rbpf->n_particles; i++)
        {
            sum_log_vol += rbpf->mu[i];
        }
        rbpf_real_t log_vol = sum_log_vol / rbpf->n_particles;
        rbpf_real_t vol = rbpf_exp(log_vol);

        mmpf->model_output[k].log_vol_mean = log_vol;
        mmpf->model_output[k].vol_mean = vol;
        mmpf->model_output[k].ess = (rbpf_real_t)rbpf->n_particles;
        mmpf->model_output[k].outlier_fraction = RBPF_REAL(0.0);

        mmpf->weighted_vol += mmpf->weights[k] * vol;
        mmpf->weighted_log_vol += mmpf->weights[k] * log_vol;
    }
    mmpf->weighted_vol_std = RBPF_REAL(0.0); /* No uncertainty at init */

    return mmpf;
}

void mmpf_destroy(MMPF_ROCKS *mmpf)
{
    if (!mmpf)
        return;

    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        if (mmpf->rbpf[k])
            rbpf_ksc_destroy(mmpf->rbpf[k]);
        if (mmpf->learner[k])
        {
            param_learn_free(mmpf->learner[k]);
            free(mmpf->learner[k]);
        }
        if (mmpf->buffer[k])
            mmpf_buffer_destroy(mmpf->buffer[k]);
        if (mmpf->mixed_buffer[k])
            mmpf_buffer_destroy(mmpf->mixed_buffer[k]);
    }

    free(mmpf);
}

void mmpf_reset(MMPF_ROCKS *mmpf, rbpf_real_t initial_vol)
{
    rbpf_real_t log_vol = rbpf_log(initial_vol);
    rbpf_real_t var0 = RBPF_REAL(0.5);

    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        rbpf_ksc_init(mmpf->rbpf[k], log_vol, var0);
        param_learn_reset(mmpf->learner[k]);
        param_learn_broadcast_priors(mmpf->learner[k]);

        /* CRITICAL: Sync initial params from Storvik → RBPF */
        mmpf_sync_parameters(mmpf->rbpf[k], mmpf->learner[k]);
    }

    /* Reset weights */
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->weights[k] = mmpf->config.initial_weights[k];
        mmpf->log_weights[k] = rbpf_log(mmpf->weights[k]);
    }
    normalize_weights(mmpf->weights, MMPF_N_MODELS);

    /* Reset cached outputs */
    mmpf->weighted_vol = initial_vol;
    mmpf->weighted_log_vol = log_vol;
    mmpf->weighted_vol_std = RBPF_REAL(0.0);
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        memset(&mmpf->model_output[k], 0, sizeof(RBPF_KSC_Output));
        mmpf->model_output[k].vol_mean = initial_vol;
        mmpf->model_output[k].log_vol_mean = log_vol;
        mmpf->model_output[k].ess = (rbpf_real_t)mmpf->n_particles;
        mmpf->model_likelihood[k] = RBPF_REAL(1.0);
    }

    /* Reset state */
    mmpf->outlier_fraction = RBPF_REAL(0.0);
    mmpf->current_stickiness = mmpf->config.base_stickiness;
    update_transition_matrix(mmpf);

    mmpf->dominant = MMPF_CALM;
    mmpf->prev_dominant = MMPF_CALM;
    mmpf->ticks_in_regime = 0;

    mmpf->total_steps = 0;
    mmpf->regime_switches = 0;
    mmpf->imm_mix_count = 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN STEP
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_step(MMPF_ROCKS *mmpf, rbpf_real_t y, MMPF_Output *output)
{

    /* 1. Update stickiness from t-1 OCSN */
    update_transition_matrix(mmpf);

    /* 2. Compute mixing weights */
    compute_mixing_weights(mmpf);
    compute_mixing_counts(mmpf);

    /* 3-5. IMM mixing step */
    imm_mixing_step(mmpf);

    /* 5.5 CRITICAL: Sync learned params from Storvik → RBPF
     *
     * imm_mixing_step() imports mixed particles into Storvik's SoA buffers,
     * but rbpf_ksc_predict() reads from RBPF's particle_mu_vol/particle_sigma_vol.
     * Without this sync, RBPF uses stale parameters and ignores all learning!
     *
     * NOTE: Disable for unit tests that verify hypothesis discrimination.
     * When sync is enabled, all models converge toward similar params.
     */
    if (mmpf->config.enable_storvik_sync)
    {
        for (int k = 0; k < MMPF_N_MODELS; k++)
        {
            mmpf_sync_parameters(mmpf->rbpf[k], mmpf->learner[k]);
        }
    }

    /* 6. Handle zero/near-zero returns (HFT CRITICAL)
     *
     * In HFT, price often doesn't move for many ticks (r_t = 0).
     * Naive handling treats r=0 as σ ≈ exp(-11.5) ≈ 0.00001, which:
     *   1. Biases the filter toward impossibly low volatility
     *   2. Causes ESS collapse when price finally ticks
     *
     * Solutions:
     *   0 = Skip update entirely (treat as censored/missing data)
     *   1 = Use configurable floor (min_log_return_sq)
     *   2 = Interval likelihood (not implemented - complex)
     */
    int skip_update = 0;
    rbpf_real_t y_log;

    if (rbpf_fabs(y) < RBPF_REAL(1e-10))
    {
        switch (mmpf->config.zero_return_policy)
        {
        case 0: /* Skip update - treat as censored data */
            skip_update = 1;
            y_log = RBPF_REAL(0.0); /* Won't be used */
            break;
        case 1: /* Use floor */
        default:
            y_log = mmpf->config.min_log_return_sq;
            break;
        }
    }
    else
    {
        y_log = rbpf_log(y * y);
        /* Also clamp to floor to prevent extremely negative values */
        if (y_log < mmpf->config.min_log_return_sq)
        {
            y_log = mmpf->config.min_log_return_sq;
        }
    }

    /* 7. Step each RBPF with ROBUST OCSN and cache outputs
     *
     * CRITICAL: We call the individual RBPF functions directly instead of
     * rbpf_ksc_step() to use the robust OCSN update path.
     *
     * rbpf_ksc_step() calls rbpf_ksc_update() which does NOT handle outliers.
     * We need rbpf_ksc_update_robust() for outlier detection + adaptive stickiness.
     */

    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        RBPF_KSC *rbpf = mmpf->rbpf[k];
        RBPF_KSC_Output *out = &mmpf->model_output[k];

        /* 7a. Regime transition */
        rbpf_ksc_transition(rbpf);

        /* 7b. Kalman predict */
        rbpf_ksc_predict(rbpf);

        rbpf_real_t marginal_lik;

        if (skip_update)
        {
            /* Skip update - just use predict-only state
             * Marginal likelihood = 1.0 (no information) */
            marginal_lik = RBPF_REAL(1.0);

            /* Copy predict to posterior (no update) */
            for (int i = 0; i < rbpf->n_particles; i++)
            {
                rbpf->mu[i] = rbpf->mu_pred[i];
                rbpf->var[i] = rbpf->var_pred[i];
            }
        }
        else
        {
            /* 7c. Robust OCSN update (with outlier component) */
            if (mmpf->robust_ocsn.enabled)
            {
                marginal_lik = rbpf_ksc_update_robust(rbpf, y_log, &mmpf->robust_ocsn);
            }
            else
            {
                marginal_lik = rbpf_ksc_update(rbpf, y_log);
            }
        }

        /* 7d. Compute outputs (before resample) */
        rbpf_ksc_compute_outputs(rbpf, marginal_lik, out);

        /* 7e. Resample if needed (skip if update was skipped - weights are uniform) */
        if (!skip_update)
        {
            out->resampled = rbpf_ksc_resample(rbpf);
        }
        else
        {
            out->resampled = 0;
        }

        /* 7f. Compute outlier fraction for this model */
        if (mmpf->robust_ocsn.enabled && !skip_update)
        {
            out->outlier_fraction = rbpf_ksc_compute_outlier_fraction(
                rbpf, y_log, &mmpf->robust_ocsn);
        }
        else
        {
            out->outlier_fraction = RBPF_REAL(0.0);
        }

        /* Cache marginal likelihood */
        mmpf->model_likelihood[k] = marginal_lik;
    }

    /* 7. IMM Bayesian weight update
     *
     * Standard IMM: π'_j = c_j × L_j / Σ_k c_k × L_k
     * Where c_j = Σ_i π_i × T_ij is the predicted prior from transition matrix.
     *
     * We need to:
     *   1. Compute predicted prior c_j from current weights and transition
     *   2. Multiply by likelihood L_j
     *   3. Normalize
     *
     * CRITICAL: Don't accumulate log-likelihoods! Compute fresh posterior each step.
     */

    /* Step 7a: Compute predicted prior c_j = Σ_i π_i × T_ij */
    rbpf_real_t c[MMPF_N_MODELS];
    for (int j = 0; j < MMPF_N_MODELS; j++)
    {
        c[j] = RBPF_REAL(0.0);
        for (int i = 0; i < MMPF_N_MODELS; i++)
        {
            c[j] += mmpf->weights[i] * mmpf->transition[i][j];
        }
        /* Floor to prevent numerical death */
        if (c[j] < RBPF_REAL(1e-6))
            c[j] = RBPF_REAL(1e-6);
    }

    /* Step 7b: Compute posterior π'_j ∝ c_j × L_j in log space */
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->log_weights[k] = rbpf_log(c[k]) + rbpf_log(mmpf->model_likelihood[k] + RBPF_EPS);
    }

    /* Step 7c: Normalize (log_weights_to_linear handles numerical stability) */
    log_weights_to_linear(mmpf->log_weights, mmpf->weights, MMPF_N_MODELS);

    /* Ensure log_weights are also normalized for diagnostics */
    rbpf_real_t lse = log_sum_exp(mmpf->log_weights, MMPF_N_MODELS);
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->log_weights[k] -= lse;
    }

    /* 8. Compute weighted outputs from cached model outputs */
    mmpf->weighted_vol = RBPF_REAL(0.0);
    mmpf->weighted_log_vol = RBPF_REAL(0.0);

    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->weighted_vol += mmpf->weights[k] * mmpf->model_output[k].vol_mean;
        mmpf->weighted_log_vol += mmpf->weights[k] * mmpf->model_output[k].log_vol_mean;
    }

    /* Compute volatility uncertainty using Law of Total Variance:
     *
     *   Var[V] = E[Var[V|M]] + Var[E[V|M]]
     *          = within_model_var + between_model_var
     *
     * Current code only computed between-model variance (model disagreement).
     * Missing within-model variance (each model's internal uncertainty).
     *
     * For Kelly sizing: f = μ / σ². Underestimating σ² leads to overbetting
     * when models agree but are individually uncertain.
     *
     * For log-normal volatility: Var[exp(h)] = exp(2μ + σ²)(exp(σ²) - 1)
     * Approximation: Var[V] ≈ V² × Var[log(V)] for small variance
     */

    /* Between-model variance: Var[E[V|M]] = Σ w_k (μ_k - μ̄)² */
    rbpf_real_t between_var = RBPF_REAL(0.0);
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        rbpf_real_t diff = mmpf->model_output[k].vol_mean - mmpf->weighted_vol;
        between_var += mmpf->weights[k] * diff * diff;
    }
    mmpf->between_model_var = between_var;

    /* Within-model variance: E[Var[V|M]] = Σ w_k × σ²_k
     *
     * Each model has log_vol_var = Var[h] from Kalman filter.
     * For V = exp(h): Var[V] ≈ V² × Var[h] (delta method approximation)
     * More precise: Var[exp(h)] = exp(2μ + σ²)(exp(σ²) - 1)
     */
    rbpf_real_t within_var = RBPF_REAL(0.0);
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        rbpf_real_t log_vol_var = mmpf->model_output[k].log_vol_var;
        rbpf_real_t vol_mean = mmpf->model_output[k].vol_mean;

        /* Delta method: Var[exp(h)] ≈ exp(h)² × Var[h] = V² × σ²_h */
        rbpf_real_t model_var = vol_mean * vol_mean * log_vol_var;

        within_var += mmpf->weights[k] * model_var;
    }
    mmpf->within_model_var = within_var;

    /* Total variance = between + within */
    rbpf_real_t total_var = between_var + within_var;
    mmpf->weighted_vol_std = rbpf_sqrt(total_var);

    /* 9. Determine dominant model and compute weighted outlier fraction */
    int dom = argmax(mmpf->weights, MMPF_N_MODELS);
    mmpf->prev_dominant = mmpf->dominant;
    mmpf->dominant = (MMPF_Hypothesis)dom;

    /* Track regime stability */
    if (mmpf->dominant == mmpf->prev_dominant)
    {
        mmpf->ticks_in_regime++;
    }
    else
    {
        mmpf->ticks_in_regime = 1;
        mmpf->regime_switches++;
    }

    /* Compute WEIGHTED outlier fraction across ALL models for adaptive stickiness
     * This is more robust than using only dominant model's outlier fraction,
     * since dominant can switch during regime transitions. */
    mmpf->outlier_fraction = RBPF_REAL(0.0);
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->outlier_fraction += mmpf->weights[k] * mmpf->model_output[k].outlier_fraction;
    }

    mmpf->total_steps++;

    /* Fill output structure if provided */
    if (output)
    {
        output->volatility = mmpf->weighted_vol;
        output->log_volatility = mmpf->weighted_log_vol;
        output->volatility_std = mmpf->weighted_vol_std;
        output->between_model_var = between_var;
        output->within_model_var = within_var;
        output->update_skipped = skip_update;

        for (int k = 0; k < MMPF_N_MODELS; k++)
        {
            output->weights[k] = mmpf->weights[k];
            output->model_vol[k] = mmpf->model_output[k].vol_mean;
            output->model_log_vol[k] = mmpf->model_output[k].log_vol_mean;
            output->model_log_vol_var[k] = mmpf->model_output[k].log_vol_var;
            output->model_likelihood[k] = mmpf->model_likelihood[k];
            output->model_ess[k] = mmpf->model_output[k].ess;
        }

        output->dominant = mmpf->dominant;
        output->dominant_prob = mmpf->weights[dom];

        output->outlier_fraction = mmpf->outlier_fraction;
        output->current_stickiness = mmpf->current_stickiness;

        for (int i = 0; i < MMPF_N_MODELS; i++)
        {
            for (int j = 0; j < MMPF_N_MODELS; j++)
            {
                output->mixing_weights[i][j] = mmpf->mixing_weights[i][j];
            }
        }

        output->regime_stable = (mmpf->ticks_in_regime >= 10) ? 1 : 0;
        output->ticks_in_regime = mmpf->ticks_in_regime;
    }
}

void mmpf_step_apf(MMPF_ROCKS *mmpf, rbpf_real_t y_current, rbpf_real_t y_next,
                   MMPF_Output *output)
{
    /* TODO: Implement APF variant */
    /* For now, fall back to standard step */
    (void)y_next;
    mmpf_step(mmpf, y_current, output);
}

/*═══════════════════════════════════════════════════════════════════════════
 * OUTPUT ACCESSORS (Read from cached values - NO filter calls)
 *═══════════════════════════════════════════════════════════════════════════*/

rbpf_real_t mmpf_get_volatility(const MMPF_ROCKS *mmpf)
{
    return mmpf->weighted_vol;
}

rbpf_real_t mmpf_get_log_volatility(const MMPF_ROCKS *mmpf)
{
    return mmpf->weighted_log_vol;
}

rbpf_real_t mmpf_get_volatility_std(const MMPF_ROCKS *mmpf)
{
    return mmpf->weighted_vol_std;
}

MMPF_Hypothesis mmpf_get_dominant(const MMPF_ROCKS *mmpf)
{
    return mmpf->dominant;
}

rbpf_real_t mmpf_get_dominant_probability(const MMPF_ROCKS *mmpf)
{
    return mmpf->weights[mmpf->dominant];
}

void mmpf_get_weights(const MMPF_ROCKS *mmpf, rbpf_real_t *weights)
{
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        weights[k] = mmpf->weights[k];
    }
}

rbpf_real_t mmpf_get_outlier_fraction(const MMPF_ROCKS *mmpf)
{
    return mmpf->outlier_fraction;
}

rbpf_real_t mmpf_get_stickiness(const MMPF_ROCKS *mmpf)
{
    return mmpf->current_stickiness;
}

rbpf_real_t mmpf_get_model_volatility(const MMPF_ROCKS *mmpf, MMPF_Hypothesis model)
{
    return mmpf->model_output[model].vol_mean;
}

rbpf_real_t mmpf_get_model_ess(const MMPF_ROCKS *mmpf, MMPF_Hypothesis model)
{
    return mmpf->model_output[model].ess;
}

const RBPF_KSC *mmpf_get_rbpf(const MMPF_ROCKS *mmpf, MMPF_Hypothesis model)
{
    return mmpf->rbpf[model];
}

const ParamLearner *mmpf_get_learner(const MMPF_ROCKS *mmpf, MMPF_Hypothesis model)
{
    return mmpf->learner[model];
}

/*═══════════════════════════════════════════════════════════════════════════
 * IMM CONTROL
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_set_transition_matrix(MMPF_ROCKS *mmpf, const rbpf_real_t *transition)
{
    for (int i = 0; i < MMPF_N_MODELS; i++)
    {
        for (int j = 0; j < MMPF_N_MODELS; j++)
        {
            mmpf->transition[i][j] = transition[i * MMPF_N_MODELS + j];
        }
    }
}

void mmpf_set_stickiness(MMPF_ROCKS *mmpf, rbpf_real_t base, rbpf_real_t min_s)
{
    mmpf->config.base_stickiness = base;
    mmpf->config.min_stickiness = min_s;
}

void mmpf_set_adaptive_stickiness(MMPF_ROCKS *mmpf, int enable)
{
    mmpf->config.enable_adaptive_stickiness = enable;
}

void mmpf_set_weights(MMPF_ROCKS *mmpf, const rbpf_real_t *weights)
{
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->weights[k] = weights[k];
    }
    normalize_weights(mmpf->weights, MMPF_N_MODELS);

    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->log_weights[k] = rbpf_log(mmpf->weights[k]);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_print_summary(const MMPF_ROCKS *mmpf)
{
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("MMPF-ROCKS Summary\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("Particles per model: %d\n", mmpf->n_particles);
    printf("Total steps:         %lu\n", (unsigned long)mmpf->total_steps);
    printf("Regime switches:     %lu\n", (unsigned long)mmpf->regime_switches);
    printf("IMM mix count:       %lu\n", (unsigned long)mmpf->imm_mix_count);
    printf("\n");
    printf("Model Weights:\n");
    printf("  Calm:   %.4f\n", (double)mmpf->weights[MMPF_CALM]);
    printf("  Trend:  %.4f\n", (double)mmpf->weights[MMPF_TREND]);
    printf("  Crisis: %.4f\n", (double)mmpf->weights[MMPF_CRISIS]);
    printf("\n");
    printf("Dominant: %s (prob=%.4f)\n",
           mmpf->dominant == MMPF_CALM ? "Calm" : mmpf->dominant == MMPF_TREND ? "Trend"
                                                                               : "Crisis",
           (double)mmpf->weights[mmpf->dominant]);
    printf("Ticks in regime: %d\n", mmpf->ticks_in_regime);
    printf("\n");
    printf("Weighted volatility: %.6f\n", (double)mmpf->weighted_vol);
    printf("Outlier fraction:    %.4f\n", (double)mmpf->outlier_fraction);
    printf("Current stickiness:  %.4f\n", (double)mmpf->current_stickiness);
    printf("═══════════════════════════════════════════════════════════════════\n");
}

void mmpf_print_output(const MMPF_Output *output)
{
    printf("MMPF Output:\n");
    printf("  Volatility:      %.6f (std=%.6f)\n",
           (double)output->volatility, (double)output->volatility_std);
    printf("  Log-volatility:  %.6f\n", (double)output->log_volatility);
    printf("  Weights:         [%.4f, %.4f, %.4f]\n",
           (double)output->weights[0], (double)output->weights[1], (double)output->weights[2]);
    printf("  Dominant:        %d (prob=%.4f)\n",
           output->dominant, (double)output->dominant_prob);
    printf("  Outlier frac:    %.4f\n", (double)output->outlier_fraction);
    printf("  Stickiness:      %.4f\n", (double)output->current_stickiness);
    printf("  Regime stable:   %d (ticks=%d)\n",
           output->regime_stable, output->ticks_in_regime);
}

void mmpf_get_diagnostics(const MMPF_ROCKS *mmpf,
                          uint64_t *total_steps,
                          uint64_t *regime_switches,
                          uint64_t *imm_mix_count)
{
    if (total_steps)
        *total_steps = mmpf->total_steps;
    if (regime_switches)
        *regime_switches = mmpf->regime_switches;
    if (imm_mix_count)
        *imm_mix_count = mmpf->imm_mix_count;
}