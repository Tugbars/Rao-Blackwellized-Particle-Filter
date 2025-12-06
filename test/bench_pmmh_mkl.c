/**
 * @file bench_pmmh_mkl.c
 * @brief Benchmark: Scalar PMMH vs MKL-optimized PMMH
 *
 * Compile (MKL):
 *   source /opt/intel/oneapi/setvars.sh
 *   gcc -O3 -march=native -fopenmp -DUSE_MKL \
 *       -I${MKLROOT}/include bench_pmmh_mkl.c \
 *       -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed \
 *       -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core \
 *       -lgomp -lpthread -lm -ldl -o bench_pmmh_mkl
 *
 * Compile (scalar fallback):
 *   gcc -O3 -march=native bench_pmmh_mkl.c -lm -o bench_pmmh_scalar
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#ifdef USE_MKL
#include "pmmh_mkl.h"
#define HAS_MKL 1
/* omp_get_wtime and omp_get_max_threads come from omp.h via pmmh_mkl.h */
#else
#define HAS_MKL 0
/* Portable high-resolution timer fallback (no OpenMP) */
#ifdef _WIN32
#include <windows.h>
static double get_time_sec(void)
{
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}
#else
static double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif
#define omp_get_wtime get_time_sec
static int omp_get_max_threads(void) { return 1; }
#endif

/*============================================================================
 * SCALAR IMPLEMENTATION (for comparison / fallback)
 *============================================================================*/

static uint64_t scalar_rng_s[2] = {12345678901234567ULL, 98765432109876543ULL};

static void scalar_rng_seed(uint64_t seed)
{
    scalar_rng_s[0] = seed;
    scalar_rng_s[1] = seed ^ 0xDEADBEEFCAFEBABEULL;
}

static inline double scalar_randu(void)
{
    uint64_t s1 = scalar_rng_s[0];
    const uint64_t s0 = scalar_rng_s[1];
    scalar_rng_s[0] = s0;
    s1 ^= s1 << 23;
    scalar_rng_s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return ((scalar_rng_s[1] + s0) >> 11) * (1.0 / 9007199254740992.0);
}

static inline double scalar_randn(void)
{
    double u1 = scalar_randu(), u2 = scalar_randu();
    if (u1 < 1e-10)
        u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

#ifndef USE_MKL
typedef struct
{
    double drift;
    double mu_vol;
    double sigma_vol;
} PMMHParams;

typedef struct
{
    PMMHParams mean;
    PMMHParams std;
} PMMHPrior;

typedef struct
{
    PMMHParams posterior_mean;
    PMMHParams posterior_std;
    double acceptance_rate;
    int n_samples;
    double elapsed_ms;
} PMMHResult;
#endif

static double scalar_pmmh_log_lik(const double *ret, int n, int np,
                                  const PMMHParams *p, double theta)
{
    double *lv = (double *)malloc(np * sizeof(double));
    double *lv2 = (double *)malloc(np * sizeof(double));
    double *w = (double *)malloc(np * sizeof(double));

    for (int i = 0; i < np; i++)
        lv[i] = p->mu_vol + scalar_randn() * 0.3;

    double ll = 0;
    double one_m_theta = 1.0 - theta;
    double theta_mu = theta * p->mu_vol;

    for (int t = 0; t < n; t++)
    {
        double max_lw = -1e30;
        for (int i = 0; i < np; i++)
        {
            lv2[i] = one_m_theta * lv[i] + theta_mu + p->sigma_vol * scalar_randn();
            double vol = exp(lv2[i]);
            double z = (ret[t] - p->drift) / vol;
            w[i] = -0.5 * z * z - lv2[i];
            if (w[i] > max_lw)
                max_lw = w[i];
        }

        double sum_w = 0;
        for (int i = 0; i < np; i++)
        {
            w[i] = exp(w[i] - max_lw);
            sum_w += w[i];
        }
        ll += max_lw + log(sum_w / np);

        /* Systematic resampling */
        double u = scalar_randu() / np;
        double cumsum = 0;
        int j = 0;
        for (int i = 0; i < np; i++)
        {
            double target = u + (double)i / np;
            while (cumsum + w[j] / sum_w < target && j < np - 1)
            {
                cumsum += w[j] / sum_w;
                j++;
            }
            lv[i] = lv2[j];
        }
    }

    free(lv);
    free(lv2);
    free(w);
    return ll;
}

static void scalar_pmmh_run(const double *ret, int n, int np,
                            const PMMHPrior *prior, double theta,
                            int n_iter, int n_burn, PMMHResult *res)
{
    double t_start = omp_get_wtime();

    PMMHParams cur = prior->mean;
    double cur_ll = scalar_pmmh_log_lik(ret, n, np, &cur, theta);

    int n_acc = 0;
    double sum_d = 0, sum_m = 0, sum_s = 0;
    int n_samples = 0;

    for (int it = 0; it < n_iter; it++)
    {
        PMMHParams prop;
        prop.drift = cur.drift + scalar_randn() * 0.0015;
        prop.mu_vol = cur.mu_vol + scalar_randn() * 0.25;
        prop.sigma_vol = cur.sigma_vol * exp(scalar_randn() * 0.15);

        prop.drift = fmax(-0.01, fmin(0.01, prop.drift));
        prop.mu_vol = fmax(-8, fmin(0, prop.mu_vol));
        prop.sigma_vol = fmax(0.01, fmin(0.5, prop.sigma_vol));

        double prop_ll = scalar_pmmh_log_lik(ret, n, np, &prop, theta);

        /* Simple prior (just bounds check) */
        double log_alpha = prop_ll - cur_ll;

        if (log(scalar_randu()) < log_alpha)
        {
            cur = prop;
            cur_ll = prop_ll;
            n_acc++;
        }

        if (it >= n_burn)
        {
            sum_d += cur.drift;
            sum_m += cur.mu_vol;
            sum_s += cur.sigma_vol;
            n_samples++;
        }
    }

    res->posterior_mean.drift = sum_d / n_samples;
    res->posterior_mean.mu_vol = sum_m / n_samples;
    res->posterior_mean.sigma_vol = sum_s / n_samples;
    res->acceptance_rate = (double)n_acc / n_iter;
    res->n_samples = n_samples;
    res->elapsed_ms = (omp_get_wtime() - t_start) * 1000.0;
}

/*============================================================================
 * DATA GENERATION
 *============================================================================*/

static void generate_sv_data(double *returns, int n,
                             double drift, double mu_vol, double sigma_vol,
                             double theta)
{
    double log_vol = mu_vol;

    for (int t = 0; t < n; t++)
    {
        log_vol = (1.0 - theta) * log_vol + theta * mu_vol + sigma_vol * scalar_randn();
        double vol = exp(log_vol);
        returns[t] = drift + vol * scalar_randn();
    }
}

/*============================================================================
 * MAIN BENCHMARK
 *============================================================================*/

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv; /* Unused */

#if HAS_MKL
    /* CRITICAL: MKL internal ops (vdExp, vdRng) should be single-threaded.
     * Vector sizes (256-512) are too small to benefit from threading.
     * Parallelism happens at chain level via OpenMP. */
    mkl_set_num_threads(1);
    mkl_set_dynamic(0);

    /* OpenMP threads for parallel chains */
    omp_set_num_threads(16);

    printf("MKL: sequential (small vectors)\n");
    printf("OMP: 16 threads (parallel chains)\n\n");
#endif

    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║           PMMH BENCHMARK: Scalar vs MKL                           ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║ MKL:        %s                                                   ║\n", HAS_MKL ? "ENABLED" : "DISABLED");
    printf("║ OMP Threads: %-3d                                                  ║\n", omp_get_max_threads());
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    /* Configuration - optimal: 256p/300i */
    int n_obs = 200;
    int n_iter = 300;
    int n_burn = 100;
    int n_runs = 5;

    /* True parameters */
    double true_drift = 0.0005;
    double true_mu_vol = -3.0;
    double true_sigma_vol = 0.10;
    double theta_vol = 0.02;

    /* Generate data */
    double *returns = (double *)malloc(n_obs * sizeof(double));
    scalar_rng_seed(42);
    generate_sv_data(returns, n_obs, true_drift, true_mu_vol, true_sigma_vol, theta_vol);

    /* Prior - same as tune_pmmh.c */
    PMMHPrior prior = {
        .mean = {.drift = 0.0, .mu_vol = -3.5, .sigma_vol = 0.15},
        .std = {.drift = 0.002, .mu_vol = 1.5, .sigma_vol = 0.15}};

    printf("Data: %d observations, true params: drift=%.4f, μ_v=%.2f, σ_v=%.3f\n\n",
           n_obs, true_drift, true_mu_vol, true_sigma_vol);

    /* Test different particle counts */
    int particle_counts[] = {64, 128, 256, 512};
    int n_particle_configs = sizeof(particle_counts) / sizeof(particle_counts[0]);

    printf("┌──────────┬──────────────────────────────────────────────────────────┐\n");
    printf("│ Particles│   Scalar Time   │    MKL Time    │ Speedup │ μ_v Error  │\n");
    printf("├──────────┼─────────────────┼────────────────┼─────────┼────────────┤\n");

    for (int pc = 0; pc < n_particle_configs; pc++)
    {
        int np = particle_counts[pc];

        /* Benchmark scalar */
        double scalar_total_ms = 0;
        double scalar_mu_err = 0;
        PMMHResult scalar_res;

        for (int run = 0; run < n_runs; run++)
        {
            scalar_rng_seed(12345 + run);
            scalar_pmmh_run(returns, n_obs, np, &prior, theta_vol, n_iter, n_burn, &scalar_res);
            scalar_total_ms += scalar_res.elapsed_ms;
            scalar_mu_err += fabs(scalar_res.posterior_mean.mu_vol - true_mu_vol);
        }
        scalar_total_ms /= n_runs;
        scalar_mu_err /= n_runs;

#if HAS_MKL
        /* Benchmark MKL */
        double mkl_total_ms = 0;
        double mkl_mu_err = 0;
        PMMHResult mkl_res;

        for (int run = 0; run < n_runs; run++)
        {
            pmmh_run_mkl(returns, n_obs, &prior, theta_vol, n_iter, n_burn, np,
                         12345 + run, &mkl_res);
            mkl_total_ms += mkl_res.elapsed_ms;
            mkl_mu_err += fabs(mkl_res.posterior_mean.mu_vol - true_mu_vol);
        }
        mkl_total_ms /= n_runs;
        mkl_mu_err /= n_runs;

        double speedup = scalar_total_ms / mkl_total_ms;

        printf("│ %8d │ %10.1f ms   │ %10.1f ms  │  %5.2fx │   %.3f    │\n",
               np, scalar_total_ms, mkl_total_ms, speedup, mkl_mu_err);
#else
        printf("│ %8d │ %10.1f ms   │      N/A       │   N/A   │   %.3f    │\n",
               np, scalar_total_ms, scalar_mu_err);
#endif
    }

    printf("└──────────┴─────────────────┴────────────────┴─────────┴────────────┘\n");

#if HAS_MKL
    /* Test parallel chains - use all P-core threads */
    int n_chains = 16; /* 8 P-cores × 2 HT = 16 threads */

    printf("\n┌──────────────────────────────────────────────────────────────────┐\n");
    printf("│ PARALLEL CHAINS TEST (256 particles, %d chains)                  │\n", n_chains);
    printf("├──────────────────────────────────────────────────────────────────┤\n");

    PMMHParallelResult par_res;
    pmmh_run_parallel(returns, n_obs, &prior, theta_vol, n_iter, n_burn, 256, n_chains, &par_res);

    PMMHResult agg;
    pmmh_parallel_aggregate(&par_res, &agg);

    printf("│ Total time:      %8.1f ms                                     │\n", par_res.total_elapsed_ms);
    printf("│ Per-chain avg:   %8.1f ms                                     │\n", par_res.total_elapsed_ms / n_chains);
    printf("│ Posterior μ_v:   %8.3f (true: %.3f, err: %.3f)               │\n",
           agg.posterior_mean.mu_vol, true_mu_vol,
           fabs(agg.posterior_mean.mu_vol - true_mu_vol));
    printf("│ Cross-chain std: %8.4f                                        │\n", agg.posterior_std.mu_vol);
    printf("│ Acceptance rate: %8.1f%%                                       │\n", agg.acceptance_rate * 100);
    printf("│ Effective samples: %d (16 chains × %d post-burnin)             │\n",
           agg.n_samples, (n_iter - n_burn));
    printf("└──────────────────────────────────────────────────────────────────┘\n");

    pmmh_parallel_free(&par_res);
#endif

    free(returns);

    printf("\nDone.\n");
    return 0;
}