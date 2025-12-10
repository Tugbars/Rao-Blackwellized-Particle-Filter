/*=============================================================================
 * RBPF-APF: Auxiliary Particle Filter Extension (OPTIMIZED)
 *
 * Lookahead-based resampling for improved regime change detection.
 * Uses y_{t+1} to bias resampling toward "promising" particles.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * OPTIMIZATIONS APPLIED
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * 1. MKL VML: Batch vsLn/vsExp instead of scalar log/exp
 * 2. MKL VSL: ICDF-based Gaussian generation (vdRngGaussian)
 * 3. POINTER SWAPPING: Double buffering eliminates memcpy in resample
 * 4. LOOP FUSION: Predict + likelihood in single pass for cache locality
 * 5. SoA LAYOUT: Preserved for optimal SIMD auto-vectorization
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * INDEX-BASED RESAMPLING (for Storvik Integration)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * When using Storvik parameter learning, the per-particle parameter arrays
 * must be resampled consistently with the RBPF state arrays. This requires:
 *
 *   1. Compute resample indices (which source particle each new particle copies)
 *   2. Apply indices to RBPF arrays (mu, var, regime)
 *   3. Apply SAME indices to Storvik arrays (external to this file)
 *
 * New API:
 *   - rbpf_ksc_apf_compute_indices(): Compute lookahead weights + indices
 *   - rbpf_ksc_apf_apply_indices(): Apply indices to RBPF arrays
 *   - rbpf_ksc_apf_get_indices(): Get last computed indices (for Storvik)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Author: RBPF-KSC Project
 * License: MIT
 *===========================================================================*/

#include "rbpf_ksc.h"
#include <string.h>

/*─────────────────────────────────────────────────────────────────────────────
 * APF CONFIGURATION
 *───────────────────────────────────────────────────────────────────────────*/

#define APF_SURPRISE_THRESHOLD RBPF_REAL(3.0)
#define APF_VOL_RATIO_THRESHOLD RBPF_REAL(1.5)
#define APF_VARIANCE_INFLATION RBPF_REAL(2.5)
#define APF_BLEND_ALPHA RBPF_REAL(0.8)

/*─────────────────────────────────────────────────────────────────────────────
 * OMORI MIXTURE CONSTANTS (Shotgun components: 2, 7, 9)
 *───────────────────────────────────────────────────────────────────────────*/

static const rbpf_real_t APF_SHOTGUN_MEAN[3] = {
    RBPF_REAL(0.73504),  /* Component 2: Peak */
    RBPF_REAL(-5.55246), /* Component 7: Left tail */
    RBPF_REAL(-14.65000) /* Component 9: Extreme */
};

static const rbpf_real_t APF_SHOTGUN_VAR[3] = {
    RBPF_REAL(0.26768),
    RBPF_REAL(2.54498),
    RBPF_REAL(7.33342)};

static const rbpf_real_t APF_SHOTGUN_LOG_PROB[3] = {
    RBPF_REAL(-2.036), /* log(0.131) */
    RBPF_REAL(-2.884), /* log(0.056) */
    RBPF_REAL(-6.768)  /* log(0.00115) */
};

#define APF_N_SHOTGUN 3

/*─────────────────────────────────────────────────────────────────────────────
 * NOTE: Static buffer removed for thread safety.
 * Resample indices are now passed via indices_out parameter.
 *───────────────────────────────────────────────────────────────────────────*/

#define APF_MAX_PARTICLES 2048

/*─────────────────────────────────────────────────────────────────────────────
 * APF LOOKAHEAD LIKELIHOOD (FUSED PREDICT + LIKELIHOOD)
 *───────────────────────────────────────────────────────────────────────────*/

static void apf_compute_lookahead_weights_fused(
    RBPF_KSC *rbpf,
    rbpf_real_t y_next,
    rbpf_real_t *lookahead_log_weights)
{
    const int n = rbpf->n_particles;
    const int n_regimes = rbpf->n_regimes;
    const rbpf_real_t H = RBPF_REAL(2.0);
    const rbpf_real_t H2 = RBPF_REAL(4.0);
    const rbpf_real_t NEG_HALF = RBPF_REAL(-0.5);

    const rbpf_real_t *restrict mu = rbpf->mu;
    const rbpf_real_t *restrict var = rbpf->var;
    const int *restrict regime = rbpf->regime;
    const RBPF_RegimeParams *params = rbpf->params;

    /* FIX: Check if we should use per-particle learned params (Option B) */
    const int use_particles = rbpf->use_learned_params ||
                              (rbpf->liu_west.enabled &&
                               rbpf->liu_west.tick_count >= rbpf->liu_west.warmup_ticks);

    rbpf_real_t *restrict S_peak = rbpf->mu_pred;
    rbpf_real_t *restrict S_tail = rbpf->var_pred;
    rbpf_real_t *restrict S_extreme = rbpf->scratch1;
    rbpf_real_t *restrict log_S_best = rbpf->scratch2;

    /* Pass 1: Predict + Compute S values */
    for (int i = 0; i < n; i++)
    {
        int r = regime[i];
        rbpf_real_t theta_r = params[r].theta;
        rbpf_real_t omt = RBPF_REAL(1.0) - theta_r;

        /* FIX: Select correct parameter source */
        rbpf_real_t mu_r, q_r;

        if (use_particles && rbpf->particle_mu_vol && rbpf->particle_sigma_vol)
        {
            int idx = i * n_regimes + r;
            mu_r = rbpf->particle_mu_vol[idx];
            rbpf_real_t sigma = rbpf->particle_sigma_vol[idx];
            q_r = sigma * sigma;
        }
        else
        {
            mu_r = params[r].mu_vol;
            q_r = params[r].q;
        }

        rbpf_real_t mu_pred = mu_r + omt * (mu[i] - mu_r);
        rbpf_real_t var_pred = omt * omt * var[i] + q_r * APF_VARIANCE_INFLATION;
        rbpf_real_t H2_var = H2 * var_pred;
        rbpf_real_t H_mu = H * mu_pred;

        S_peak[i] = H2_var + APF_SHOTGUN_VAR[0];
        S_tail[i] = H2_var + APF_SHOTGUN_VAR[1];
        S_extreme[i] = H2_var + APF_SHOTGUN_VAR[2];

        rbpf_real_t res0 = y_next - APF_SHOTGUN_MEAN[0] - H_mu;
        rbpf_real_t res1 = y_next - APF_SHOTGUN_MEAN[1] - H_mu;
        rbpf_real_t res2 = y_next - APF_SHOTGUN_MEAN[2] - H_mu;

        rbpf_real_t pll0 = APF_SHOTGUN_LOG_PROB[0] + NEG_HALF * res0 * res0 / S_peak[i];
        rbpf_real_t pll1 = APF_SHOTGUN_LOG_PROB[1] + NEG_HALF * res1 * res1 / S_tail[i];
        rbpf_real_t pll2 = APF_SHOTGUN_LOG_PROB[2] + NEG_HALF * res2 * res2 / S_extreme[i];

        if (pll0 >= pll1 && pll0 >= pll2)
        {
            log_S_best[i] = S_peak[i];
            lookahead_log_weights[i] = pll0;
        }
        else if (pll1 >= pll2)
        {
            log_S_best[i] = S_tail[i];
            lookahead_log_weights[i] = pll1;
        }
        else
        {
            log_S_best[i] = S_extreme[i];
            lookahead_log_weights[i] = pll2;
        }
    }

    /* Pass 2: Batch log(S) using MKL VML */
    rbpf_vsLn(n, log_S_best, log_S_best);

    for (int i = 0; i < n; i++)
    {
        lookahead_log_weights[i] += NEG_HALF * log_S_best[i];
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * APF COMBINED WEIGHTS
 *───────────────────────────────────────────────────────────────────────────*/

static void apf_combine_weights(
    const rbpf_real_t *log_weight_current,
    const rbpf_real_t *lookahead_log_weight,
    rbpf_real_t *log_weight_combined,
    int n)
{
    memcpy(log_weight_combined, log_weight_current, n * sizeof(rbpf_real_t));
    rbpf_cblas_axpy(n, APF_BLEND_ALPHA, lookahead_log_weight, 1, log_weight_combined, 1);
}

/*─────────────────────────────────────────────────────────────────────────────
 * APF COMPUTE RESAMPLE INDICES (NEW - for Storvik Integration)
 *
 * Computes which source particle each new particle should copy from.
 * Does NOT apply the resampling - just computes the mapping.
 *
 * This allows external code to resample additional arrays (Storvik params)
 * using the same indices.
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_apf_compute_resample_indices(
    RBPF_KSC *rbpf,
    const rbpf_real_t *log_weight_combined,
    int *indices_out)
{
    const int n = rbpf->n_particles;
    rbpf_real_t *w_norm = rbpf->w_norm;
    rbpf_real_t *cdf = rbpf->cumsum;

    /* Find max for numerical stability */
    rbpf_real_t max_lw = log_weight_combined[0];
    for (int i = 1; i < n; i++)
    {
        if (log_weight_combined[i] > max_lw)
        {
            max_lw = log_weight_combined[i];
        }
    }

    /* Compute shifted log-weights */
    for (int i = 0; i < n; i++)
    {
        w_norm[i] = log_weight_combined[i] - max_lw;
    }

    /* Batch exp using MKL VML */
    rbpf_vsExp(n, w_norm, w_norm);

    /* Normalize */
    rbpf_real_t sum_w = rbpf_cblas_asum(n, w_norm, 1);
    if (sum_w < RBPF_REAL(1e-30))
        sum_w = RBPF_REAL(1.0);
    rbpf_cblas_scal(n, RBPF_REAL(1.0) / sum_w, w_norm, 1);

    /* Compute CDF */
    cdf[0] = w_norm[0];
    for (int i = 1; i < n; i++)
    {
        cdf[i] = cdf[i - 1] + w_norm[i];
    }
    cdf[n - 1] = RBPF_REAL(1.0);

    /* Systematic resampling → compute indices */
    rbpf_real_t u0 = rbpf_pcg32_uniform(&rbpf->pcg[0]) / (rbpf_real_t)n;

    int j = 0;
    for (int i = 0; i < n; i++)
    {
        rbpf_real_t u = u0 + (rbpf_real_t)i / (rbpf_real_t)n;

        while (j < n - 1 && cdf[j] < u)
        {
            j++;
        }

        indices_out[i] = j;
    }

    /* NOTE: Static buffer removed for thread safety.
     * Use rbpf_ksc_step_apf_indexed() which returns indices directly. */
}

/*─────────────────────────────────────────────────────────────────────────────
 * APF APPLY RESAMPLE INDICES (NEW - for Storvik Integration)
 *
 * Applies pre-computed indices to RBPF arrays.
 * Uses double buffering with pointer swap for efficiency.
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_apf_apply_resample_indices(
    RBPF_KSC *rbpf,
    const int *indices)
{
    const int n = rbpf->n_particles;

    rbpf_real_t *restrict mu = rbpf->mu;
    rbpf_real_t *restrict var = rbpf->var;
    int *restrict regime = rbpf->regime;

    rbpf_real_t *restrict mu_new = rbpf->mu_tmp;
    rbpf_real_t *restrict var_new = rbpf->var_tmp;
    int *restrict regime_new = rbpf->regime_tmp;

    /* Apply indices */
    for (int i = 0; i < n; i++)
    {
        int src = indices[i];
        mu_new[i] = mu[src];
        var_new[i] = var[src];
        regime_new[i] = regime[src];
    }

    /* Pointer swap */
    rbpf->mu = mu_new;
    rbpf->var = var_new;
    rbpf->regime = regime_new;

    rbpf->mu_tmp = mu;
    rbpf->var_tmp = var;
    rbpf->regime_tmp = regime;

    /* Also copy to rbpf->indices for compatibility with standard resample */
    memcpy(rbpf->indices, indices, n * sizeof(int));

    /* Reset to uniform weights */
    rbpf_real_t log_uniform = -rbpf_log((rbpf_real_t)n);
    for (int i = 0; i < n; i++)
    {
        rbpf->log_weight[i] = log_uniform;
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * APF GET LAST RESAMPLE INDICES (DEPRECATED)
 *
 * This function used a static buffer which is not thread-safe.
 * Use rbpf_ksc_step_apf_indexed() instead, which returns indices directly.
 *───────────────────────────────────────────────────────────────────────────*/

int rbpf_ksc_apf_get_resample_indices(int *indices_out, int max_n)
{
    (void)indices_out;
    (void)max_n;
    /* DEPRECATED: Static buffer removed for thread safety.
     * Always returns 0. Use rbpf_ksc_step_apf_indexed() instead. */
    return 0;
}

/*─────────────────────────────────────────────────────────────────────────────
 * LEGACY: APF RESAMPLING (combined compute + apply)
 *
 * Kept for backward compatibility with non-Storvik usage.
 *───────────────────────────────────────────────────────────────────────────*/

static void apf_resample_optimized(
    RBPF_KSC *rbpf,
    const rbpf_real_t *log_weight_combined)
{
    const int n = rbpf->n_particles;
    int indices[APF_MAX_PARTICLES];

    if (n > APF_MAX_PARTICLES)
    {
        /* Fallback for very large particle counts */
        int *dyn_indices = (int *)malloc(n * sizeof(int));
        rbpf_ksc_apf_compute_resample_indices(rbpf, log_weight_combined, dyn_indices);
        rbpf_ksc_apf_apply_resample_indices(rbpf, dyn_indices);
        free(dyn_indices);
    }
    else
    {
        rbpf_ksc_apf_compute_resample_indices(rbpf, log_weight_combined, indices);
        rbpf_ksc_apf_apply_resample_indices(rbpf, indices);
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * PUBLIC API: APF Step (Standard - no Storvik awareness)
 *
 * For backward compatibility. If using Storvik, use rbpf_ext_step_apf() instead.
 *───────────────────────────────────────────────────────────────────────────*/

/* Forward declarations */
extern void rbpf_ksc_predict(RBPF_KSC *rbpf);
extern rbpf_real_t rbpf_ksc_update(RBPF_KSC *rbpf, rbpf_real_t y);
extern void rbpf_ksc_transition(RBPF_KSC *rbpf);
extern void rbpf_ksc_compute_outputs(RBPF_KSC *rbpf, rbpf_real_t marginal, RBPF_KSC_Output *out);
extern int rbpf_ksc_resample(RBPF_KSC *rbpf);

void rbpf_ksc_step_apf(
    RBPF_KSC *rbpf,
    rbpf_real_t obs_current,
    rbpf_real_t obs_next,
    RBPF_KSC_Output *out)
{
    /* Transform observations */
    rbpf_real_t y_current, y_next;

    if (rbpf_fabs(obs_current) < RBPF_REAL(1e-10))
    {
        y_current = RBPF_REAL(-23.0);
    }
    else
    {
        y_current = rbpf_log(obs_current * obs_current);
    }

    if (rbpf_fabs(obs_next) < RBPF_REAL(1e-10))
    {
        y_next = RBPF_REAL(-23.0);
    }
    else
    {
        y_next = rbpf_log(obs_next * obs_next);
    }

    /* Workspace */
    rbpf_real_t *lookahead_log_weights = rbpf->mu_accum;
    rbpf_real_t *combined_log_weights = rbpf->var_accum;

    /* Step 1: Regime transition */
    rbpf_ksc_transition(rbpf);

    /* Step 2: Predict */
    rbpf_ksc_predict(rbpf);

    /* Step 3: Update */
    rbpf_real_t marginal = rbpf_ksc_update(rbpf, y_current);

    /* Step 4: Outputs */
    rbpf_ksc_compute_outputs(rbpf, marginal, out);

    /* Step 5: APF Lookahead */
    apf_compute_lookahead_weights_fused(rbpf, y_next, lookahead_log_weights);

    /* Step 6: Combine weights */
    apf_combine_weights(rbpf->log_weight, lookahead_log_weights,
                        combined_log_weights, rbpf->n_particles);

    /* Step 7: Resample (legacy - use index-based for Storvik) */
    apf_resample_optimized(rbpf, combined_log_weights);
    out->resampled = 1;
    out->apf_triggered = 1;

    /* Step 8: Liu-West */
    if (rbpf->liu_west.enabled)
    {
        rbpf->liu_west.tick_count++;
        for (int r = 0; r < rbpf->n_regimes; r++)
        {
            rbpf_ksc_get_learned_params(rbpf, r,
                                        &out->learned_mu_vol[r],
                                        &out->learned_sigma_vol[r]);
        }
    }
    else
    {
        for (int r = 0; r < rbpf->n_regimes; r++)
        {
            out->learned_mu_vol[r] = rbpf->params[r].mu_vol;
            out->learned_sigma_vol[r] = rbpf->params[r].sigma_vol;
        }
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * PUBLIC API: APF Step with Index Output (for Storvik Integration)
 *
 * This version returns the resample indices so external code can apply
 * them to additional arrays (Storvik parameters).
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_step_apf_indexed(
    RBPF_KSC *rbpf,
    rbpf_real_t obs_current,
    rbpf_real_t obs_next,
    RBPF_KSC_Output *out,
    int *resample_indices_out)
{
    /* Transform observations */
    rbpf_real_t y_current, y_next;

    if (rbpf_fabs(obs_current) < RBPF_REAL(1e-10))
    {
        y_current = RBPF_REAL(-23.0);
    }
    else
    {
        y_current = rbpf_log(obs_current * obs_current);
    }

    if (rbpf_fabs(obs_next) < RBPF_REAL(1e-10))
    {
        y_next = RBPF_REAL(-23.0);
    }
    else
    {
        y_next = rbpf_log(obs_next * obs_next);
    }

    /* Workspace */
    rbpf_real_t *lookahead_log_weights = rbpf->mu_accum;
    rbpf_real_t *combined_log_weights = rbpf->var_accum;

    /* Step 1: Regime transition */
    rbpf_ksc_transition(rbpf);

    /* Step 2: Predict */
    rbpf_ksc_predict(rbpf);

    /* Step 3: Update */
    rbpf_real_t marginal = rbpf_ksc_update(rbpf, y_current);

    /* Step 4: Outputs */
    rbpf_ksc_compute_outputs(rbpf, marginal, out);

    /* Step 5: APF Lookahead */
    apf_compute_lookahead_weights_fused(rbpf, y_next, lookahead_log_weights);

    /* Step 6: Combine weights */
    apf_combine_weights(rbpf->log_weight, lookahead_log_weights,
                        combined_log_weights, rbpf->n_particles);

    /* Step 7: Compute indices (NOT applied yet) */
    rbpf_ksc_apf_compute_resample_indices(rbpf, combined_log_weights, resample_indices_out);

    /* Step 8: Apply indices to RBPF arrays */
    rbpf_ksc_apf_apply_resample_indices(rbpf, resample_indices_out);

    out->resampled = 1;
    out->apf_triggered = 1;

    /* Step 9: Liu-West
     *
     * TODO: Missing jitter/mutation step!
     * When Liu-West is enabled, we should call:
     *   rbpf_ksc_liu_west_compute_stats(rbpf);
     *   rbpf_ksc_liu_west_resample(rbpf, resample_indices_out);
     *
     * This requires making those functions non-static in rbpf_ksc.c
     * and adding extern declarations here.
     *
     * For now, in Storvik mode (use_learned_params=1), liu_west.enabled=0,
     * so this code path is not taken.
     */
    if (rbpf->liu_west.enabled)
    {
        rbpf->liu_west.tick_count++;
        /* WARNING: Jitter not applied! Particles may collapse. */
        for (int r = 0; r < rbpf->n_regimes; r++)
        {
            rbpf_ksc_get_learned_params(rbpf, r,
                                        &out->learned_mu_vol[r],
                                        &out->learned_sigma_vol[r]);
        }
    }
    else
    {
        for (int r = 0; r < rbpf->n_regimes; r++)
        {
            out->learned_mu_vol[r] = rbpf->params[r].mu_vol;
            out->learned_sigma_vol[r] = rbpf->params[r].sigma_vol;
        }
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * PUBLIC API: Adaptive APF Step
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_step_adaptive(
    RBPF_KSC *rbpf,
    rbpf_real_t obs_current,
    rbpf_real_t obs_next,
    RBPF_KSC_Output *out)
{
    int use_apf = 0;

    if (obs_next != RBPF_REAL(0.0))
    {
        rbpf_real_t recent_vol_ratio = rbpf->detection.vol_ema_short /
                                       (rbpf->detection.vol_ema_long + RBPF_REAL(1e-10));

        rbpf_real_t y_current = rbpf_log(obs_current * obs_current + RBPF_REAL(1e-20));
        rbpf_real_t y_expected = RBPF_REAL(2.0) * rbpf->mu[0] + RBPF_REAL(-1.27);
        rbpf_real_t quick_surprise = rbpf_fabs(y_current - y_expected);

        if (quick_surprise > APF_SURPRISE_THRESHOLD ||
            recent_vol_ratio > APF_VOL_RATIO_THRESHOLD)
        {
            use_apf = 1;
        }

        if (rbpf_ksc_apf_forced())
        {
            use_apf = 1;
        }
    }

    if (use_apf)
    {
        rbpf_ksc_step_apf(rbpf, obs_current, obs_next, out);
    }
    else
    {
        rbpf_ksc_step(rbpf, obs_current, out);
        out->apf_triggered = 0;
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * PUBLIC API: Force APF for next N steps
 *───────────────────────────────────────────────────────────────────────────*/

static int apf_force_count = 0;

void rbpf_ksc_force_apf(int n_steps)
{
    apf_force_count = n_steps;
}

int rbpf_ksc_apf_forced(void)
{
    if (apf_force_count > 0)
    {
        apf_force_count--;
        return 1;
    }
    return 0;
}

/*─────────────────────────────────────────────────────────────────────────────
 * DIAGNOSTIC: APF Statistics
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    int total_steps;
    int apf_steps;
    rbpf_real_t avg_lookahead_entropy;
} RBPF_APF_Stats;

static RBPF_APF_Stats apf_stats = {0};

void rbpf_apf_reset_stats(void)
{
    memset(&apf_stats, 0, sizeof(apf_stats));
}

void rbpf_apf_get_stats(int *total, int *apf_count, rbpf_real_t *apf_ratio)
{
    *total = apf_stats.total_steps;
    *apf_count = apf_stats.apf_steps;
    *apf_ratio = (apf_stats.total_steps > 0)
                     ? (rbpf_real_t)apf_stats.apf_steps / apf_stats.total_steps
                     : RBPF_REAL(0.0);
}