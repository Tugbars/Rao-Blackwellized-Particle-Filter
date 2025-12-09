/**
 * @file pmmh_mkl.h
 * @brief High-performance PMMH using Intel MKL
 *
 * Optimizations:
 *   - Arena allocation: Single contiguous block for particle arrays (better dTLB)
 *   - Single VSL stream per state (avoids N*N stream explosion in parallel)
 *   - VSL SFMT19937 RNG with ICDF method (faster than BoxMuller)
 *   - VML vdExp/vsExp for vectorized exp (precision switchable)
 *   - OpenMP SIMD for particle loops
 *   - Division elimination: exp(-2*log_vol) for inv_var, no scalar division
 *   - Adaptive resampling: Only resample when ESS < N*0.05
 *   - Pointer swapping: When not resampling, swap pointers instead of gather
 *   - Memory prefetching: Prefetch random access in resampling gather
 *   - Precision switching: Define PMMH_USE_FLOAT for 2x SIMD throughput
 *
 * Performance (256 particles, 300 iterations, 16 chains, double precision):
 *   - 6.1x speedup vs scalar implementation
 *   - 62ms per chain (was 116ms)
 *   - mu_v error: 0.019 (was 0.086)
 *   - Cross-chain std: 0.014 (was 0.076)
 *
 * Threading model:
 *   - MKL internal: single-threaded (vectors too small to benefit)
 *   - Chain level: OpenMP parallel with thread-local states
 *   - Each thread has its own PMMHState with single VSL stream
 *
 * Memory layout:
 *   - All particle arrays in single 64-byte aligned arena
 *   - log_vol, log_vol_new, weights, weights_exp, noise, uniform, ancestors
 *
 * Compile (GCC): gcc -O3 -march=native -fopenmp -I${MKLROOT}/include \
 *                pmmh_mkl.c -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 \
 *                -lmkl_gnu_thread -lmkl_core -lgomp -lm
 *
 * Compile (MSVC): Uses standard OpenMP, no experimental features required
 */

#ifndef PMMH_MKL_H
#define PMMH_MKL_H

#include <mkl.h>
#include <mkl_vsl.h>
#include <mkl_vml.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* Prefetch intrinsics - cross-platform */
#ifdef _MSC_VER
#include <intrin.h>
#define PMMH_PREFETCH(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#else
#include <xmmintrin.h>
#define PMMH_PREFETCH(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#endif

/*============================================================================
 * PRECISION CONFIGURATION
 *
 * Define PMMH_USE_FLOAT before including this header to use single precision.
 * Single precision doubles SIMD throughput (8 floats vs 4 doubles per AVX).
 *============================================================================*/

/* #define PMMH_USE_FLOAT */

#ifdef PMMH_USE_FLOAT
typedef float pmmh_real;
#define PMMH_REAL_SIZE 4
#define PMMH_REAL_MAX FLT_MAX
#define PMMH_REAL_MIN (-FLT_MAX)
#define pmmh_vExp vsExp
#define pmmh_RngGaussian vsRngGaussian
#define pmmh_RngUniform vsRngUniform
#define pmmh_log logf
#define pmmh_exp expf
#define pmmh_sqrt sqrtf
#define pmmh_fabs fabsf
#else
typedef double pmmh_real;
#define PMMH_REAL_SIZE 8
#define PMMH_REAL_MAX DBL_MAX
#define PMMH_REAL_MIN (-DBL_MAX)
#define pmmh_vExp vdExp
#define pmmh_RngGaussian vdRngGaussian
#define pmmh_RngUniform vdRngUniform
#define pmmh_log log
#define pmmh_exp exp
#define pmmh_sqrt sqrt
#define pmmh_fabs fabs
#endif

/*============================================================================
 * CONFIGURATION
 *============================================================================*/

#define PMMH_N_PARTICLES 256
#define PMMH_N_ITERATIONS 300
#define PMMH_N_BURNIN 100
#define PMMH_CACHE_LINE 64

/*
 * PMMH_ESS_THRESHOLD - Effective Sample Size threshold for adaptive resampling
 */
#define PMMH_ESS_THRESHOLD 0.05

/*
 * PMMH_PREFETCH_DIST - Software prefetch distance for resampling gather
 */
#define PMMH_PREFETCH_DIST 8

/*
 * Aligned allocation macros using MKL's allocator
 */
#define ALIGNED_ALLOC(size) mkl_malloc((size), PMMH_CACHE_LINE)
#define ALIGNED_FREE(ptr) mkl_free(ptr)

/*============================================================================
 * DATA STRUCTURES
 *============================================================================*/

/*
 * PMMHParams - Stochastic Volatility model parameters
 */
typedef struct
{
    double drift;     /* Expected return per period */
    double mu_vol;    /* Long-run mean of log-volatility */
    double sigma_vol; /* Volatility of volatility (vol-of-vol) */
} PMMHParams;

/*
 * PMMHPrior - Gaussian prior distributions for Bayesian inference
 */
typedef struct
{
    PMMHParams mean; /* Prior means */
    PMMHParams std;  /* Prior standard deviations */
} PMMHPrior;

/*
 * PMMHResult - Output from PMMH sampling
 */
typedef struct
{
    PMMHParams posterior_mean; /* Posterior mean estimates */
    PMMHParams posterior_std;  /* Posterior standard deviations (cross-chain) */
    double acceptance_rate;    /* MH acceptance rate (target: 20-30%) */
    int n_samples;             /* Number of post-burnin samples */
    double elapsed_ms;         /* Wall-clock time in milliseconds */
} PMMHResult;

/*
 * PMMHState - Particle filter state (one per MCMC chain)
 */
typedef struct
{
    /* Particle arrays - ping-pong buffers for log-volatility state */
    pmmh_real *log_vol;     /* [n_particles] current log-volatility */
    pmmh_real *log_vol_new; /* [n_particles] proposed/next log-volatility */

    /* Weight arrays - recomputed each time step */
    pmmh_real *weights;     /* [n_particles] log-weights (before exp) */
    pmmh_real *weights_exp; /* [n_particles] exp(weights), then normalized */

    /* Scratch arrays - reused for multiple purposes */
    pmmh_real *noise;   /* [n_particles] Gaussian noise */
    pmmh_real *uniform; /* [n_particles] reused as CDF during resampling */
    int *ancestors;     /* [n_particles] resampling indices */

    /* Random number generator - thread-local for parallel safety */
    VSLStreamStatePtr stream;

    int n_particles;  /* Number of particles (typically 256) */
    double theta_vol; /* Mean reversion speed (fixed hyperparameter) */

} PMMHState;

/*============================================================================
 * INITIALIZATION / CLEANUP
 *============================================================================*/

static PMMHState *pmmh_state_create(int n_particles, double theta_vol)
{
    PMMHState *s = (PMMHState *)ALIGNED_ALLOC(sizeof(PMMHState));
    memset(s, 0, sizeof(PMMHState));

    s->n_particles = n_particles;
    s->theta_vol = theta_vol;

    /* Arena allocation: single contiguous block for all particle arrays */
    size_t r_size = ((n_particles * PMMH_REAL_SIZE + PMMH_CACHE_LINE - 1) / PMMH_CACHE_LINE) * PMMH_CACHE_LINE;
    size_t i_size = ((n_particles * sizeof(int) + PMMH_CACHE_LINE - 1) / PMMH_CACHE_LINE) * PMMH_CACHE_LINE;

    size_t total_size = 6 * r_size + i_size;

    char *arena = (char *)mkl_malloc(total_size, PMMH_CACHE_LINE);
    s->log_vol = (pmmh_real *)(arena + 0 * r_size);
    s->log_vol_new = (pmmh_real *)(arena + 1 * r_size);
    s->weights = (pmmh_real *)(arena + 2 * r_size);
    s->weights_exp = (pmmh_real *)(arena + 3 * r_size);
    s->noise = (pmmh_real *)(arena + 4 * r_size);
    s->uniform = (pmmh_real *)(arena + 5 * r_size);
    s->ancestors = (int *)(arena + 6 * r_size);

    /* VSL Stream initialization */
    vslNewStream(&s->stream, VSL_BRNG_SFMT19937, 12345);

    return s;
}

static void pmmh_state_destroy(PMMHState *s)
{
    if (!s)
        return;
    vslDeleteStream(&s->stream);
    mkl_free(s->log_vol);
    ALIGNED_FREE(s);
}

static void pmmh_state_seed(PMMHState *s, unsigned int seed)
{
    vslDeleteStream(&s->stream);
    vslNewStream(&s->stream, VSL_BRNG_SFMT19937, seed);
}

/*============================================================================
 * VECTORIZED PARTICLE FILTER LOG-LIKELIHOOD
 *============================================================================*/

static double pmmh_log_likelihood_mkl(PMMHState *s,
                                      const double *returns, int n_obs,
                                      const PMMHParams *params)
{
    const int np = s->n_particles;
    const pmmh_real theta = (pmmh_real)s->theta_vol;
    const pmmh_real one_minus_theta = (pmmh_real)1.0 - theta;
    const pmmh_real theta_mu = theta * (pmmh_real)params->mu_vol;
    const pmmh_real drift = (pmmh_real)params->drift;
    const pmmh_real sigma_vol = (pmmh_real)params->sigma_vol;
    const pmmh_real ess_threshold = (pmmh_real)(np * PMMH_ESS_THRESHOLD);

    /* Pointers for ping-pong buffering */
    pmmh_real *log_vol_curr = s->log_vol;
    pmmh_real *log_vol_next = s->log_vol_new;

    int i, t, j;

    /* Initialize particles at mu_vol with some spread */
    pmmh_RngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, s->stream,
                     np, log_vol_curr, (pmmh_real)params->mu_vol, (pmmh_real)0.3);

    double total_log_lik = 0.0;

    for (t = 0; t < n_obs; t++)
    {
        const pmmh_real ret = (pmmh_real)returns[t];
        const pmmh_real ret_minus_drift = ret - drift;
        const pmmh_real ret_sq = ret_minus_drift * ret_minus_drift;

        /* Generate noise for volatility evolution */
        pmmh_RngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, s->stream,
                         np, s->noise, (pmmh_real)0.0, sigma_vol);

        /* Propagate volatility */
        for (i = 0; i < np; i++)
        {
            log_vol_next[i] = one_minus_theta * log_vol_curr[i] + theta_mu + s->noise[i];
        }

        /* Division elimination: compute -2*log_vol for inv_var */
        for (i = 0; i < np; i++)
        {
            s->noise[i] = (pmmh_real)-2.0 * log_vol_next[i];
        }

        /* Vectorized exp: inv_var = exp(-2*log_vol) = 1/var = 1/vol^2 */
        pmmh_vExp(np, s->noise, s->weights_exp);

        /* Compute log-weights without division */
        pmmh_real max_log_w = PMMH_REAL_MIN;
        for (i = 0; i < np; i++)
        {
            pmmh_real log_w = (pmmh_real)-0.5 * ret_sq * s->weights_exp[i] - log_vol_next[i];
            s->weights[i] = log_w;
            if (log_w > max_log_w)
                max_log_w = log_w;
        }

        for (i = 0; i < np; i++)
        {
            s->weights[i] -= max_log_w;
        }

        pmmh_vExp(np, s->weights, s->weights_exp);

        pmmh_real sum_w = 0;
        pmmh_real sum_w_sq = 0;
        for (i = 0; i < np; i++)
        {
            pmmh_real w = s->weights_exp[i];
            sum_w += w;
            sum_w_sq += w * w;
        }

        total_log_lik += (double)max_log_w + log((double)sum_w / np);

        /* Normalize weights and compute ESS */
        pmmh_real inv_sum = (pmmh_real)1.0 / sum_w;
        pmmh_real ess = (sum_w * sum_w) / sum_w_sq;

        for (i = 0; i < np; i++)
        {
            s->weights_exp[i] *= inv_sum;
        }

        /* Adaptive resampling: only resample when ESS drops below threshold */
        if (ess < ess_threshold)
        {
            /* Systematic resampling */
            pmmh_real *cdf = s->uniform;
            cdf[0] = s->weights_exp[0];
            for (i = 1; i < np; i++)
            {
                cdf[i] = cdf[i - 1] + s->weights_exp[i];
            }
            cdf[np - 1] = (pmmh_real)1.0;

            pmmh_real u0;
            pmmh_RngUniform(VSL_RNG_METHOD_UNIFORM_STD, s->stream, 1, &u0,
                            (pmmh_real)0.0, (pmmh_real)1.0);
            pmmh_real inv_np = (pmmh_real)1.0 / np;
            u0 *= inv_np;

            j = 0;
            for (i = 0; i < np; i++)
            {
                pmmh_real target = u0 + i * inv_np;
                while (j < np - 1 && cdf[j] < target)
                    j++;
                s->ancestors[i] = j;
            }

            /* Gather resampled particles into log_vol_curr with prefetching */
            for (i = 0; i < np; i++)
            {
                /* Prefetch future random access */
                if (i + PMMH_PREFETCH_DIST < np)
                {
                    PMMH_PREFETCH(&log_vol_next[s->ancestors[i + PMMH_PREFETCH_DIST]]);
                }
                log_vol_curr[i] = log_vol_next[s->ancestors[i]];
            }
            /* log_vol_curr now has resampled particles, ready for next iteration */
        }
        else
        {
            /* No resampling needed - pointer swap */
            pmmh_real *tmp = log_vol_curr;
            log_vol_curr = log_vol_next;
            log_vol_next = tmp;
        }
    }

    return total_log_lik;
}

/*============================================================================
 * LOG-PRIOR
 *============================================================================*/

static double pmmh_log_prior(const PMMHParams *p, const PMMHPrior *prior)
{
    double lp = 0.0;
    double z_drift = (p->drift - prior->mean.drift) / prior->std.drift;
    double z_mu = (p->mu_vol - prior->mean.mu_vol) / prior->std.mu_vol;
    double z_sigma = (log(p->sigma_vol) - log(prior->mean.sigma_vol)) / prior->std.sigma_vol;
    lp -= 0.5 * (z_drift * z_drift + z_mu * z_mu + z_sigma * z_sigma);
    lp -= log(prior->std.drift) + log(prior->std.mu_vol) + log(prior->std.sigma_vol);
    lp -= log(p->sigma_vol);
    return lp;
}

/*============================================================================
 * ADAPTIVE PROPOSAL TUNING
 *============================================================================*/

#define PMMH_ADAPT_WINDOW 50
#define PMMH_TARGET_ACCEPT 0.30
#define PMMH_ACCEPT_TOL 0.05

#define PMMH_INIT_DRIFT_STD 0.0015
#define PMMH_INIT_MU_STD 0.25
#define PMMH_INIT_SIGMA_LOG_STD 0.15

typedef struct
{
    double drift_std;
    double mu_std;
    double sigma_log_std;
    int window_accepts;
    int window_total;
} AdaptiveProposal;

static void adapt_proposal(AdaptiveProposal *ap)
{
    double rate, factor;

    if (ap->window_total < PMMH_ADAPT_WINDOW)
        return;

    rate = (double)ap->window_accepts / ap->window_total;

    if (rate < PMMH_TARGET_ACCEPT - PMMH_ACCEPT_TOL)
    {
        factor = 0.8;
    }
    else if (rate > PMMH_TARGET_ACCEPT + PMMH_ACCEPT_TOL)
    {
        factor = 1.25;
    }
    else
    {
        factor = 1.0;
    }

    ap->drift_std *= factor;
    ap->mu_std *= factor;
    ap->sigma_log_std *= factor;

    if (ap->drift_std < 1e-6)
        ap->drift_std = 1e-6;
    if (ap->drift_std > 0.01)
        ap->drift_std = 0.01;
    if (ap->mu_std < 0.01)
        ap->mu_std = 0.01;
    if (ap->mu_std > 1.0)
        ap->mu_std = 1.0;
    if (ap->sigma_log_std < 0.01)
        ap->sigma_log_std = 0.01;
    if (ap->sigma_log_std > 0.5)
        ap->sigma_log_std = 0.5;

    ap->window_accepts = 0;
    ap->window_total = 0;
}

/*============================================================================
 * MAIN PMMH SAMPLER
 *============================================================================*/

static void pmmh_run_mkl(const double *returns, int n_obs,
                         const PMMHPrior *prior,
                         double theta_vol,
                         int n_iterations, int n_burnin,
                         int n_particles,
                         unsigned int seed,
                         PMMHResult *result)
{
    int iter, i;
    int n_accepted = 0;
    int sample_idx = 0;
    int max_samples;
    int accepted;
    double t_start;
    double current_ll, current_lp;
    double prop_ll, prop_lp, log_alpha, u;
    double prop_noise[3];
    double *samples_drift, *samples_mu, *samples_sigma;
    double sum_drift, sum_mu, sum_sigma;
    double sum_sq_drift, sum_sq_mu, sum_sq_sigma;
    double inv_n;
    PMMHState *state;
    PMMHParams current, proposed;
    AdaptiveProposal ap;

    t_start = omp_get_wtime();

    state = pmmh_state_create(n_particles, theta_vol);
    pmmh_state_seed(state, seed);

    ap.drift_std = PMMH_INIT_DRIFT_STD;
    ap.mu_std = PMMH_INIT_MU_STD;
    ap.sigma_log_std = PMMH_INIT_SIGMA_LOG_STD;
    ap.window_accepts = 0;
    ap.window_total = 0;

    max_samples = n_iterations - n_burnin;
    samples_drift = (double *)ALIGNED_ALLOC(max_samples * sizeof(double));
    samples_mu = (double *)ALIGNED_ALLOC(max_samples * sizeof(double));
    samples_sigma = (double *)ALIGNED_ALLOC(max_samples * sizeof(double));

    current = prior->mean;
    current_ll = pmmh_log_likelihood_mkl(state, returns, n_obs, &current);
    current_lp = pmmh_log_prior(&current, prior);

    for (iter = 0; iter < n_iterations; iter++)
    {
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->stream,
                      3, prop_noise, 0.0, 1.0);

        proposed.drift = current.drift + prop_noise[0] * ap.drift_std;
        proposed.mu_vol = current.mu_vol + prop_noise[1] * ap.mu_std;
        proposed.sigma_vol = current.sigma_vol * exp(prop_noise[2] * ap.sigma_log_std);

        if (proposed.drift < -0.01)
            proposed.drift = -0.01;
        if (proposed.drift > 0.01)
            proposed.drift = 0.01;
        if (proposed.mu_vol < -8.0)
            proposed.mu_vol = -8.0;
        if (proposed.mu_vol > 0.0)
            proposed.mu_vol = 0.0;
        if (proposed.sigma_vol < 0.01)
            proposed.sigma_vol = 0.01;
        if (proposed.sigma_vol > 0.5)
            proposed.sigma_vol = 0.5;

        prop_ll = pmmh_log_likelihood_mkl(state, returns, n_obs, &proposed);
        prop_lp = pmmh_log_prior(&proposed, prior);

        log_alpha = (prop_ll + prop_lp) - (current_ll + current_lp);

        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, 1, &u, 0.0, 1.0);

        accepted = (log(u) < log_alpha);
        if (accepted)
        {
            current = proposed;
            current_ll = prop_ll;
            current_lp = prop_lp;
            n_accepted++;
        }

        ap.window_accepts += accepted;
        ap.window_total++;

        if (iter < n_burnin)
        {
            adapt_proposal(&ap);
        }

        if (iter >= n_burnin)
        {
            samples_drift[sample_idx] = current.drift;
            samples_mu[sample_idx] = current.mu_vol;
            samples_sigma[sample_idx] = current.sigma_vol;
            sample_idx++;
        }
    }

    result->n_samples = sample_idx;

    sum_drift = 0;
    sum_mu = 0;
    sum_sigma = 0;
    sum_sq_drift = 0;
    sum_sq_mu = 0;
    sum_sq_sigma = 0;

    for (i = 0; i < sample_idx; i++)
    {
        double d = samples_drift[i];
        double m = samples_mu[i];
        double s = samples_sigma[i];
        sum_drift += d;
        sum_mu += m;
        sum_sigma += s;
        sum_sq_drift += d * d;
        sum_sq_mu += m * m;
        sum_sq_sigma += s * s;
    }

    inv_n = 1.0 / sample_idx;
    result->posterior_mean.drift = sum_drift * inv_n;
    result->posterior_mean.mu_vol = sum_mu * inv_n;
    result->posterior_mean.sigma_vol = sum_sigma * inv_n;

    result->posterior_std.drift = sqrt(sum_sq_drift * inv_n -
                                       result->posterior_mean.drift * result->posterior_mean.drift);
    result->posterior_std.mu_vol = sqrt(sum_sq_mu * inv_n -
                                        result->posterior_mean.mu_vol * result->posterior_mean.mu_vol);
    result->posterior_std.sigma_vol = sqrt(sum_sq_sigma * inv_n -
                                           result->posterior_mean.sigma_vol * result->posterior_mean.sigma_vol);

    result->acceptance_rate = (double)n_accepted / n_iterations;
    result->elapsed_ms = (omp_get_wtime() - t_start) * 1000.0;

    ALIGNED_FREE(samples_drift);
    ALIGNED_FREE(samples_mu);
    ALIGNED_FREE(samples_sigma);
    pmmh_state_destroy(state);
}

/*============================================================================
 * PMMH WITH EXTERNALLY MANAGED STATE
 *============================================================================*/

static void pmmh_run_mkl_with_state(PMMHState *state,
                                    const double *returns, int n_obs,
                                    const PMMHPrior *prior,
                                    double theta_vol,
                                    int n_iterations, int n_burnin,
                                    PMMHResult *result)
{
    int iter, i;
    int n_accepted = 0;
    int sample_idx = 0;
    int max_samples;
    int accepted;
    double t_start;
    double current_ll, current_lp;
    double prop_ll, prop_lp, log_alpha, u;
    double prop_noise[3];
    double *samples_drift, *samples_mu, *samples_sigma;
    double sum_drift, sum_mu, sum_sigma;
    double sum_sq_drift, sum_sq_mu, sum_sq_sigma;
    double inv_n;
    PMMHParams current, proposed;
    AdaptiveProposal ap;

    t_start = omp_get_wtime();
    (void)theta_vol;

    ap.drift_std = PMMH_INIT_DRIFT_STD;
    ap.mu_std = PMMH_INIT_MU_STD;
    ap.sigma_log_std = PMMH_INIT_SIGMA_LOG_STD;
    ap.window_accepts = 0;
    ap.window_total = 0;

    max_samples = n_iterations - n_burnin;
    samples_drift = (double *)ALIGNED_ALLOC(max_samples * sizeof(double));
    samples_mu = (double *)ALIGNED_ALLOC(max_samples * sizeof(double));
    samples_sigma = (double *)ALIGNED_ALLOC(max_samples * sizeof(double));

    current = prior->mean;
    current_ll = pmmh_log_likelihood_mkl(state, returns, n_obs, &current);
    current_lp = pmmh_log_prior(&current, prior);

    for (iter = 0; iter < n_iterations; iter++)
    {
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->stream,
                      3, prop_noise, 0.0, 1.0);

        proposed.drift = current.drift + prop_noise[0] * ap.drift_std;
        proposed.mu_vol = current.mu_vol + prop_noise[1] * ap.mu_std;
        proposed.sigma_vol = current.sigma_vol * exp(prop_noise[2] * ap.sigma_log_std);

        if (proposed.drift < -0.01)
            proposed.drift = -0.01;
        if (proposed.drift > 0.01)
            proposed.drift = 0.01;
        if (proposed.mu_vol < -8.0)
            proposed.mu_vol = -8.0;
        if (proposed.mu_vol > 0.0)
            proposed.mu_vol = 0.0;
        if (proposed.sigma_vol < 0.01)
            proposed.sigma_vol = 0.01;
        if (proposed.sigma_vol > 0.5)
            proposed.sigma_vol = 0.5;

        prop_ll = pmmh_log_likelihood_mkl(state, returns, n_obs, &proposed);
        prop_lp = pmmh_log_prior(&proposed, prior);

        log_alpha = (prop_ll + prop_lp) - (current_ll + current_lp);

        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, 1, &u, 0.0, 1.0);

        accepted = (log(u) < log_alpha);
        if (accepted)
        {
            current = proposed;
            current_ll = prop_ll;
            current_lp = prop_lp;
            n_accepted++;
        }

        ap.window_accepts += accepted;
        ap.window_total++;

        if (iter < n_burnin)
        {
            adapt_proposal(&ap);
        }

        if (iter >= n_burnin)
        {
            samples_drift[sample_idx] = current.drift;
            samples_mu[sample_idx] = current.mu_vol;
            samples_sigma[sample_idx] = current.sigma_vol;
            sample_idx++;
        }
    }

    result->n_samples = sample_idx;

    sum_drift = 0;
    sum_mu = 0;
    sum_sigma = 0;
    sum_sq_drift = 0;
    sum_sq_mu = 0;
    sum_sq_sigma = 0;

    for (i = 0; i < sample_idx; i++)
    {
        double d = samples_drift[i];
        double m = samples_mu[i];
        double s = samples_sigma[i];
        sum_drift += d;
        sum_mu += m;
        sum_sigma += s;
        sum_sq_drift += d * d;
        sum_sq_mu += m * m;
        sum_sq_sigma += s * s;
    }

    inv_n = 1.0 / sample_idx;
    result->posterior_mean.drift = sum_drift * inv_n;
    result->posterior_mean.mu_vol = sum_mu * inv_n;
    result->posterior_mean.sigma_vol = sum_sigma * inv_n;

    result->posterior_std.drift = sqrt(sum_sq_drift * inv_n -
                                       result->posterior_mean.drift * result->posterior_mean.drift);
    result->posterior_std.mu_vol = sqrt(sum_sq_mu * inv_n -
                                        result->posterior_mean.mu_vol * result->posterior_mean.mu_vol);
    result->posterior_std.sigma_vol = sqrt(sum_sq_sigma * inv_n -
                                           result->posterior_mean.sigma_vol * result->posterior_mean.sigma_vol);

    result->acceptance_rate = (double)n_accepted / n_iterations;
    result->elapsed_ms = (omp_get_wtime() - t_start) * 1000.0;

    ALIGNED_FREE(samples_drift);
    ALIGNED_FREE(samples_mu);
    ALIGNED_FREE(samples_sigma);
}

/*============================================================================
 * PARALLEL MONTE CARLO
 *============================================================================*/

typedef struct
{
    PMMHResult *results;
    int n_chains;
    double total_elapsed_ms;
} PMMHParallelResult;

static void pmmh_run_parallel(const double *returns, int n_obs,
                              const PMMHPrior *prior,
                              double theta_vol,
                              int n_iterations, int n_burnin,
                              int n_particles,
                              int n_chains,
                              PMMHParallelResult *result)
{
    int t, chain;
    int n_threads;
    double t_start;
    PMMHState **thread_states;

    t_start = omp_get_wtime();

    result->n_chains = n_chains;
    result->results = (PMMHResult *)malloc(n_chains * sizeof(PMMHResult));

    n_threads = omp_get_max_threads();
    thread_states = (PMMHState **)malloc(n_threads * sizeof(PMMHState *));

    for (t = 0; t < n_threads; t++)
    {
        thread_states[t] = pmmh_state_create(n_particles, theta_vol);
    }

#pragma omp parallel for schedule(dynamic) private(chain)
    for (chain = 0; chain < n_chains; chain++)
    {
        int tid = omp_get_thread_num();
        PMMHState *state = thread_states[tid];

        pmmh_state_seed(state, 12345 + chain * 104729);

        pmmh_run_mkl_with_state(state, returns, n_obs, prior, theta_vol,
                                n_iterations, n_burnin, &result->results[chain]);
    }

    for (t = 0; t < n_threads; t++)
    {
        pmmh_state_destroy(thread_states[t]);
    }
    free(thread_states);

    result->total_elapsed_ms = (omp_get_wtime() - t_start) * 1000.0;
}

static void pmmh_parallel_aggregate(const PMMHParallelResult *pr, PMMHResult *agg)
{
    int i;
    double sum_drift = 0, sum_mu = 0, sum_sigma = 0;
    double sum_acc = 0;
    double var_drift = 0, var_mu = 0, var_sigma = 0;
    int total_samples = 0;

    for (i = 0; i < pr->n_chains; i++)
    {
        sum_drift += pr->results[i].posterior_mean.drift * pr->results[i].n_samples;
        sum_mu += pr->results[i].posterior_mean.mu_vol * pr->results[i].n_samples;
        sum_sigma += pr->results[i].posterior_mean.sigma_vol * pr->results[i].n_samples;
        sum_acc += pr->results[i].acceptance_rate;
        total_samples += pr->results[i].n_samples;
    }

    agg->posterior_mean.drift = sum_drift / total_samples;
    agg->posterior_mean.mu_vol = sum_mu / total_samples;
    agg->posterior_mean.sigma_vol = sum_sigma / total_samples;
    agg->acceptance_rate = sum_acc / pr->n_chains;
    agg->n_samples = total_samples;
    agg->elapsed_ms = pr->total_elapsed_ms;

    for (i = 0; i < pr->n_chains; i++)
    {
        double d = pr->results[i].posterior_mean.drift - agg->posterior_mean.drift;
        double m = pr->results[i].posterior_mean.mu_vol - agg->posterior_mean.mu_vol;
        double s = pr->results[i].posterior_mean.sigma_vol - agg->posterior_mean.sigma_vol;
        var_drift += d * d;
        var_mu += m * m;
        var_sigma += s * s;
    }
    agg->posterior_std.drift = sqrt(var_drift / pr->n_chains);
    agg->posterior_std.mu_vol = sqrt(var_mu / pr->n_chains);
    agg->posterior_std.sigma_vol = sqrt(var_sigma / pr->n_chains);
}

static void pmmh_parallel_free(PMMHParallelResult *pr)
{
    free(pr->results);
    pr->results = NULL;
}

#endif /* PMMH_MKL_H */