/* example_usage.c
 * Particle filter example and benchmark
 */
#include "particle_filter.h"
#include <stdio.h>
#include <time.h>
#include <math.h>

/* Cross-platform high-resolution timer */
#ifdef _WIN32
#include <windows.h>
static double get_time_us(void)
{
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1e6 / (double)freq.QuadPart;
}
#else
#include <sys/time.h>
static double get_time_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}
#endif

int main(int argc, char **argv)
{
    int n_particles = 4000;
    int n_iterations = 10000;

    /* Parse args */
    if (argc > 1)
        n_particles = atoi(argv[1]);
    if (argc > 2)
        n_iterations = atoi(argv[2]);

    printf("========================================\n");
    printf("Particle Filter Benchmark\n");
    printf("========================================\n\n");

    /* Create filter */
    ParticleFilter *pf = pf_create(n_particles, 4);
    if (!pf)
    {
        fprintf(stderr, "Failed to create particle filter\n");
        return 1;
    }

    pf_initialize(pf, (pf_real)100.0, (pf_real)1.0);

    /* Enable PCG RNG (faster for small N) - already enabled by default */
    pf_enable_pcg(pf, 1);

    /* Set adaptive resampling baseline volatility */
    pf_set_resample_adaptive(pf, (pf_real)0.01);

    pf_print_config(pf);

    /* Setup SSA features and precompute */
    SSAFeatures ssa = {
        .eigentriples = {1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005},
        .trend = (pf_real)0.001,
        .volatility = (pf_real)0.5};
    pf_precompute(pf, &ssa);

    /* Setup regime probs with precomputed cumulative */
    RegimeProbs rp;
    pf_real probs[4] = {0.4, 0.3, 0.2, 0.1};
    pf_set_regime_probs(&rp, probs, 4);
    pf_build_regime_lut(pf, &rp); /* Build O(1) lookup table */

    /* Warmup */
    printf("\nWarming up...\n");
    for (int i = 0; i < 100; i++)
    {
        pf_real obs = (pf_real)(100.0 + 0.01 * (i % 100));
        pf_update(pf, obs, &rp);
    }

    /* Benchmark */
    printf("Running benchmark (%d iterations)...\n", n_iterations);

    double start = get_time_us();

    PFOutput out;
    for (int i = 0; i < n_iterations; i++)
    {
        pf_real obs = (pf_real)(100.0 + 0.01 * (i % 100));
        out = pf_update(pf, obs, &rp);
    }

    double end = get_time_us();
    double total_us = end - start;
    double per_iter_us = total_us / n_iterations;

    printf("\n========================================\n");
    printf("Results\n");
    printf("========================================\n");
    printf("  Particles:      %d\n", n_particles);
    printf("  Iterations:     %d\n", n_iterations);
    printf("  Total time:     %.2f ms\n", total_us / 1000.0);
    printf("  Per iteration:  %.2f us\n", per_iter_us);
    printf("  Throughput:     %.0f updates/sec\n", 1e6 / per_iter_us);
    printf("========================================\n");

    /* Print last output */
    printf("\nLast update output:\n");
    printf("  Mean:      %.4f\n", (double)out.mean);
    printf("  Variance:  %.6f\n", (double)out.variance);
    printf("  Std dev:   %.4f\n", sqrt((double)out.variance));
    printf("  ESS:       %.1f (%.1f%%)\n", (double)out.ess, 100.0 * out.ess / n_particles);
    printf("  Resampled: %s\n", out.resampled ? "yes" : "no");
    printf("  Regime distribution:\n");
    for (int r = 0; r < 4; r++)
    {
        printf("    Regime %d: %.1f%%\n", r, 100.0 * out.regime_probs[r]);
    }

    /* Latency analysis */
    printf("\n========================================\n");
    printf("Latency Analysis (for trading)\n");
    printf("========================================\n");
    printf("  Your latency:      %.1f us\n", per_iter_us);
    printf("  ES peak tick gap:  200 us\n");
    printf("  Headroom:          %.1fx\n", 200.0 / per_iter_us);
    printf("  Status:            %s\n",
           per_iter_us < 200 ? "OK - catches all ticks" : "WARNING - may miss peaks");
    printf("========================================\n");

    /* Cleanup */
    pf_destroy(pf);

    return 0;
}