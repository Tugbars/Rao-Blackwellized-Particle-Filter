/**
 * @file tune_pmmh.c
 * @brief Tuning benchmark for PMMH - find optimal particles/iterations
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#define HAS_MKL 1
#include "pmmh_mkl.h"

/*============================================================================
 * DATA GENERATION - Same RNG as bench_pmmh_mkl.c for fair comparison
 *============================================================================*/

static uint64_t scalar_rng_s[2] = {0x12345678, 0x87654321};

static void scalar_rng_seed(uint64_t seed)
{
    scalar_rng_s[0] = seed;
    scalar_rng_s[1] = seed ^ 0xDEADBEEFCAFEBABEULL;
}

static inline double randu(void)
{
    uint64_t s1 = scalar_rng_s[0];
    const uint64_t s0 = scalar_rng_s[1];
    scalar_rng_s[0] = s0;
    s1 ^= s1 << 23;
    scalar_rng_s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return ((scalar_rng_s[1] + s0) >> 11) * (1.0 / 9007199254740992.0);
}

static double randn(void)
{
    double u1 = randu(), u2 = randu();
    if (u1 < 1e-10)
        u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

static void generate_sv_data(double *returns, int n,
                             double drift, double mu_vol, double sigma_vol,
                             double theta)
{
    double log_vol = mu_vol;
    for (int t = 0; t < n; t++)
    {
        log_vol = (1.0 - theta) * log_vol + theta * mu_vol + sigma_vol * randn();
        double vol = exp(log_vol);
        returns[t] = drift + vol * randn();
    }
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(void)
{
    /* Configure MKL */
    mkl_set_num_threads(1);
    mkl_set_dynamic(0);
    omp_set_num_threads(16);

    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║           PMMH TUNING: Particles vs Iterations                    ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    /* True parameters */
    int n_obs = 200;
    double true_drift = 0.0005;
    double true_mu_vol = -3.0;
    double true_sigma_vol = 0.10;
    double theta_vol = 0.02;

    /* Generate data - use same seed as bench_pmmh_mkl.c */
    double *returns = (double *)malloc(n_obs * sizeof(double));
    scalar_rng_seed(42);
    generate_sv_data(returns, n_obs, true_drift, true_mu_vol, true_sigma_vol, theta_vol);

    /* Prior */
    PMMHPrior prior = {
        .mean = {.drift = 0.0, .mu_vol = -3.5, .sigma_vol = 0.15},
        .std = {.drift = 0.002, .mu_vol = 1.5, .sigma_vol = 0.15}};

    /* Test configurations - focus on accuracy */
    int particles[] = {128, 256, 512, 1024};
    int iterations[] = {300, 500, 800, 1200};
    int n_particles_configs = sizeof(particles) / sizeof(particles[0]);
    int n_iter_configs = sizeof(iterations) / sizeof(iterations[0]);

    int n_runs = 5; /* Runs per config for averaging */

    printf("Testing %d particle configs × %d iteration configs × %d runs\n\n",
           n_particles_configs, n_iter_configs, n_runs);

    printf("┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐\n");
    printf("│ Particles│   Iter   │  Burnin  │ Time(ms) │ μ_v Err  │ Accept%%  │\n");
    printf("├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤\n");

    for (int pi = 0; pi < n_particles_configs; pi++)
    {
        for (int ii = 0; ii < n_iter_configs; ii++)
        {
            int np = particles[pi];
            int n_iter = iterations[ii];
            int n_burn = n_iter / 3;

            double total_time = 0;
            double total_error = 0;
            double total_accept = 0;

            for (int run = 0; run < n_runs; run++)
            {
                PMMHResult result;
                pmmh_run_mkl(returns, n_obs, &prior, theta_vol,
                             n_iter, n_burn, np,
                             12345 + run * 7919, &result);

                total_time += result.elapsed_ms;
                total_error += fabs(result.posterior_mean.mu_vol - true_mu_vol);
                total_accept += result.acceptance_rate;
            }

            double avg_time = total_time / n_runs;
            double avg_error = total_error / n_runs;
            double avg_accept = total_accept / n_runs;

            printf("│ %8d │ %8d │ %8d │ %8.1f │ %8.3f │ %8.1f │\n",
                   np, n_iter, n_burn, avg_time, avg_error, avg_accept * 100);
        }
        printf("├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤\n");
    }

    printf("\n");

    /* Now test parallel chains with best configs */
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║           PARALLEL CHAINS (16 chains)                             ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    printf("┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐\n");
    printf("│ Particles│   Iter   │ Tot(ms)  │ μ_v Err  │ X-chain σ│ Accept%%  │\n");
    printf("├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤\n");

    int best_particles[] = {256, 512, 1024};
    int best_iters[] = {300, 500};

    for (int pi = 0; pi < 3; pi++)
    {
        for (int ii = 0; ii < 2; ii++)
        {
            int np = best_particles[pi];
            int n_iter = best_iters[ii];
            int n_burn = n_iter / 3;

            PMMHParallelResult par_res;
            pmmh_run_parallel(returns, n_obs, &prior, theta_vol,
                              n_iter, n_burn, np, 16, &par_res);

            PMMHResult agg;
            pmmh_parallel_aggregate(&par_res, &agg);

            double error = fabs(agg.posterior_mean.mu_vol - true_mu_vol);

            printf("│ %8d │ %8d │ %8.1f │ %8.3f │ %8.4f │ %8.1f │\n",
                   np, n_iter, par_res.total_elapsed_ms, error,
                   agg.posterior_std.mu_vol, agg.acceptance_rate * 100);

            pmmh_parallel_free(&par_res);
        }
    }

    printf("└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘\n");

    free(returns);

    printf("\nDone.\n");
    return 0;
}