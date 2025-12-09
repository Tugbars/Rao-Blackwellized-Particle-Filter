/**
 * @file rbpf_ksc_param_integration.c
 * @brief Integration: RBPF-KSC + Sleeping Storvik Parameter Learning
 *
 * Key integration points:
 *   1. After RBPF update: Extract particle info → Storvik update
 *   2. After resample:    Sync Storvik ancestor indices
 *   3. Before predict:    Push learned params to RBPF (if Storvik mode)
 */

#include "rbpf_ksc_param_integration.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/*═══════════════════════════════════════════════════════════════════════════
 * PLATFORM-SPECIFIC TIMING
 *═══════════════════════════════════════════════════════════════════════════*/

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static double get_time_us(void)
{
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0)
    {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1e6 / (double)freq.QuadPart;
}
#else
#include <time.h>

static double get_time_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * COMPATIBILITY SHIM FOR OPTION B
 *
 * Option B requires a new field `use_learned_params` in RBPF_KSC and a
 * setter function. If your rbpf_ksc.h hasn't been patched yet, this shim
 * provides a fallback that directly sets the field.
 *
 * REQUIRED PATCH: Add to RBPF_KSC struct in rbpf_ksc.h:
 *   int use_learned_params;  // 1 = predict reads particle_mu/sigma_vol arrays
 *
 * See rbpf_ksc_option_b.patch for the complete patch.
 *═══════════════════════════════════════════════════════════════════════════*/

#ifndef RBPF_KSC_HAS_LEARNED_PARAMS_MODE
/* Fallback: directly set the field if setter doesn't exist yet */
static inline void rbpf_ksc_set_learned_params_mode(RBPF_KSC *rbpf, int enable)
{
    rbpf->use_learned_params = enable;
}
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * FORWARD DECLARATIONS (static functions used before definition)
 *═══════════════════════════════════════════════════════════════════════════*/

static void sync_storvik_to_rbpf(RBPF_Extended *ext);

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

RBPF_Extended *rbpf_ext_create(int n_particles, int n_regimes, RBPF_ParamMode mode)
{
    RBPF_Extended *ext = (RBPF_Extended *)calloc(1, sizeof(RBPF_Extended));
    if (!ext)
        return NULL;

    ext->param_mode = mode;

    /* Create core RBPF-KSC */
    ext->rbpf = rbpf_ksc_create(n_particles, n_regimes);
    if (!ext->rbpf)
    {
        free(ext);
        return NULL;
    }

    /* Allocate workspace */
    ext->particle_info = (ParticleInfo *)calloc(n_particles, sizeof(ParticleInfo));
    ext->prev_regime = (int *)calloc(n_particles, sizeof(int));
    ext->ell_lag_buffer = (rbpf_real_t *)calloc(n_particles, sizeof(rbpf_real_t));

    if (!ext->particle_info || !ext->prev_regime || !ext->ell_lag_buffer)
    {
        rbpf_ext_destroy(ext);
        return NULL;
    }

    /* Initialize Storvik if needed */
    if (mode == RBPF_PARAM_STORVIK || mode == RBPF_PARAM_HYBRID)
    {
        ParamLearnConfig cfg = param_learn_config_defaults();
        /* defaults() now returns always-awake (all intervals = 1) */

        cfg.sample_on_regime_change = true;
        cfg.sample_on_structural_break = true;
        cfg.sample_after_resampling = true;

        if (param_learn_init(&ext->storvik, &cfg, n_particles, n_regimes) != 0)
        {
            rbpf_ext_destroy(ext);
            return NULL;
        }
        ext->storvik_initialized = 1;
    }

    /*═══════════════════════════════════════════════════════════════════════
     * TRANSITION MATRIX LEARNING (Disabled by default)
     *═══════════════════════════════════════════════════════════════════════*/
    ext->trans_learn_enabled = 0;
    ext->trans_forgetting = 0.995;
    ext->trans_prior_diag = 50.0;
    ext->trans_prior_off = 1.0;
    ext->trans_update_interval = 100;
    ext->trans_ticks_since_update = 0;

    /* Zero-init transition counts (calloc already did this, but be explicit) */
    for (int i = 0; i < RBPF_MAX_REGIMES; i++)
    {
        for (int j = 0; j < RBPF_MAX_REGIMES; j++)
        {
            ext->trans_counts[i][j] = 0.0;
        }
    }

    /*═══════════════════════════════════════════════════════════════════════
     * CONFIGURE PER-PARTICLE PARAMETER MODE
     *
     * Option B: Decouple "reading per-particle arrays" from "Liu-West logic"
     *
     * use_learned_params: Controls whether predict() reads particle_mu_vol[]
     * liu_west.enabled:   Controls whether Liu-West update/resample runs
     *
     * STORVIK mode:  use_learned_params=1, liu_west.enabled=0
     *   - Predict reads particle arrays (populated by Storvik via memcpy)
     *   - Liu-West resample logic does NOT run (no wasted work)
     *   - Storvik maintains per-particle parameter diversity
     *
     * LIU_WEST mode: use_learned_params=1, liu_west.enabled=1
     *   - Standard Liu-West behavior (original)
     *
     * HYBRID mode:   use_learned_params=1, liu_west.enabled=1
     *   - Liu-West runs, then Storvik overwrites (for comparison/validation)
     *═══════════════════════════════════════════════════════════════════════*/
    if (mode == RBPF_PARAM_STORVIK)
    {
        /* Enable per-particle param reading, but NOT Liu-West logic */
        rbpf_ksc_set_learned_params_mode(ext->rbpf, 1);
        /* liu_west.enabled stays 0 (default) - no shrinkage/jitter */
    }
    else if (mode == RBPF_PARAM_LIU_WEST || mode == RBPF_PARAM_HYBRID)
    {
        /* Standard Liu-West (sets both use_learned_params=1 and enabled=1) */
        rbpf_ksc_enable_liu_west(ext->rbpf, 0.98f, 100);
    }

    return ext;
}

void rbpf_ext_destroy(RBPF_Extended *ext)
{
    if (!ext)
        return;

    if (ext->rbpf)
    {
        rbpf_ksc_destroy(ext->rbpf);
    }

    if (ext->storvik_initialized)
    {
        param_learn_free(&ext->storvik);
    }

    free(ext->particle_info);
    free(ext->prev_regime);
    free(ext->ell_lag_buffer);
    free(ext);
}

void rbpf_ext_init(RBPF_Extended *ext, rbpf_real_t mu0, rbpf_real_t var0)
{
    if (!ext)
        return;

    /* Initialize RBPF state */
    rbpf_ksc_init(ext->rbpf, mu0, var0);

    /* Initialize lag buffers */
    int n = ext->rbpf->n_particles;
    for (int i = 0; i < n; i++)
    {
        ext->ell_lag_buffer[i] = mu0;
        ext->prev_regime[i] = ext->rbpf->regime[i];
    }

    /* Initialize Storvik priors to match RBPF params */
    if (ext->storvik_initialized)
    {
        int nr = ext->rbpf->n_regimes;
        for (int r = 0; r < nr; r++)
        {
            const RBPF_RegimeParams *p = &ext->rbpf->params[r];
            /* CRITICAL: Convert θ (mean reversion) → φ (persistence) */
            rbpf_real_t phi = RBPF_REAL(1.0) - p->theta;
            param_learn_set_prior(&ext->storvik, r, p->mu_vol, phi, p->sigma_vol);
        }
        param_learn_broadcast_priors(&ext->storvik);

        /* CRITICAL: Sync Storvik mu_cached → RBPF particle_mu_vol
         * Without this, first step sees uninitialized arrays → NaN */
        sync_storvik_to_rbpf(ext);
    }

    ext->structural_break_signaled = 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_set_regime_params(RBPF_Extended *ext, int regime,
                                rbpf_real_t theta, rbpf_real_t mu_vol, rbpf_real_t sigma_vol)
{
    if (!ext || regime < 0 || regime >= RBPF_MAX_REGIMES)
        return;

    /* Set in RBPF */
    rbpf_ksc_set_regime_params(ext->rbpf, regime, theta, mu_vol, sigma_vol);

    /* Set in Storvik
     * CRITICAL: Storvik uses φ (persistence), RBPF uses θ (mean reversion)
     * Relationship: φ = 1 - θ
     */
    if (ext->storvik_initialized)
    {
        rbpf_real_t phi = RBPF_REAL(1.0) - theta;
        param_learn_set_prior(&ext->storvik, regime, mu_vol, phi, sigma_vol);
    }
}

void rbpf_ext_build_transition_lut(RBPF_Extended *ext, const rbpf_real_t *trans_matrix)
{
    if (!ext)
        return;
    rbpf_ksc_build_transition_lut(ext->rbpf, trans_matrix);
}

void rbpf_ext_set_storvik_interval(RBPF_Extended *ext, int regime, int interval)
{
    if (!ext || !ext->storvik_initialized)
        return;
    if (regime < 0 || regime >= PARAM_LEARN_MAX_REGIMES)
        return;

    ext->storvik.config.sample_interval[regime] = interval;
}

void rbpf_ext_set_hft_mode(RBPF_Extended *ext, int enable)
{
    if (!ext || !ext->storvik_initialized)
        return;

    if (enable)
    {
        /* HFT: Sleeping mode for lower latency (~28μs) */
        ext->storvik.config.sample_interval[0] = 100; /* R0: every 100 ticks */
        ext->storvik.config.sample_interval[1] = 50;  /* R1: every 50 ticks */
        ext->storvik.config.sample_interval[2] = 20;  /* R2: every 20 ticks */
        ext->storvik.config.sample_interval[3] = 5;   /* R3: every 5 ticks */
    }
    else
    {
        /* Standard: Always-awake for best tracking (~45μs) */
        ext->storvik.config.sample_interval[0] = 1;
        ext->storvik.config.sample_interval[1] = 1;
        ext->storvik.config.sample_interval[2] = 1;
        ext->storvik.config.sample_interval[3] = 1;
    }
}

void rbpf_ext_signal_structural_break(RBPF_Extended *ext)
{
    if (!ext)
        return;

    ext->structural_break_signaled = 1;

    if (ext->storvik_initialized)
    {
        param_learn_signal_structural_break(&ext->storvik);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * TRANSITION MATRIX LEARNING - PUBLIC API
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_enable_transition_learning(RBPF_Extended *ext, int enable)
{
    if (!ext)
        return;
    ext->trans_learn_enabled = enable;

    if (enable)
    {
        /* Reset counts when enabling */
        rbpf_ext_reset_transition_counts(ext);
    }
}

void rbpf_ext_configure_transition_learning(RBPF_Extended *ext,
                                            double forgetting,
                                            double prior_diag,
                                            double prior_off,
                                            int update_interval)
{
    if (!ext)
        return;

    ext->trans_forgetting = forgetting;
    ext->trans_prior_diag = prior_diag;
    ext->trans_prior_off = prior_off;
    ext->trans_update_interval = update_interval;
}

void rbpf_ext_reset_transition_counts(RBPF_Extended *ext)
{
    if (!ext)
        return;

    for (int i = 0; i < RBPF_MAX_REGIMES; i++)
    {
        for (int j = 0; j < RBPF_MAX_REGIMES; j++)
        {
            ext->trans_counts[i][j] = 0.0;
        }
    }
    ext->trans_ticks_since_update = 0;
}

double rbpf_ext_get_transition_prob(const RBPF_Extended *ext, int from, int to)
{
    if (!ext || !ext->rbpf)
        return 0.0;
    if (from < 0 || from >= ext->rbpf->n_regimes)
        return 0.0;
    if (to < 0 || to >= ext->rbpf->n_regimes)
        return 0.0;

    /* Compute current probability from counts + priors */
    int nr = ext->rbpf->n_regimes;
    double prior = (from == to) ? ext->trans_prior_diag : ext->trans_prior_off;

    double row_sum = 0.0;
    for (int j = 0; j < nr; j++)
    {
        double p = (from == j) ? ext->trans_prior_diag : ext->trans_prior_off;
        row_sum += ext->trans_counts[from][j] + p;
    }

    return (ext->trans_counts[from][to] + prior) / row_sum;
}

/*═══════════════════════════════════════════════════════════════════════════
 * INTERNAL: Extract particle info for Storvik
 *
 * CRITICAL: When resampled=true, rbpf->mu[i] is a CHILD of rbpf->indices[i].
 * We must look up the PARENT's lag value, not index i's lag value.
 *═══════════════════════════════════════════════════════════════════════════*/

static void extract_particle_info(RBPF_Extended *ext, int resampled)
{
    RBPF_KSC *rbpf = ext->rbpf;
    int n = rbpf->n_particles;

    /* Normalize weights for Storvik */
    rbpf_real_t max_lw = rbpf->log_weight[0];
    for (int i = 1; i < n; i++)
    {
        if (rbpf->log_weight[i] > max_lw)
            max_lw = rbpf->log_weight[i];
    }

    rbpf_real_t sum_w = RBPF_REAL(0.0);
    for (int i = 0; i < n; i++)
    {
        ext->particle_info[i].weight = rbpf_exp(rbpf->log_weight[i] - max_lw);
        sum_w += ext->particle_info[i].weight;
    }

    rbpf_real_t inv_sum = RBPF_REAL(1.0) / (sum_w + RBPF_REAL(1e-30));

    for (int i = 0; i < n; i++)
    {
        ParticleInfo *p = &ext->particle_info[i];

        p->regime = rbpf->regime[i];
        p->ell = rbpf->mu[i]; /* Current log-vol (already shuffled) */

        /* LINEAGE FIX: Look up correct parent after resampling */
        int parent_idx = resampled ? rbpf->indices[i] : i;
        p->ell_lag = ext->ell_lag_buffer[parent_idx];
        p->prev_regime = ext->prev_regime[parent_idx];

        p->weight *= inv_sum; /* Normalize */
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * INTERNAL: Sync Storvik learned params → RBPF particle arrays
 *
 * Element-by-element copy with type conversion:
 *   param_real (double) → rbpf_real_t (float)
 *
 * This is safe because Storvik initializes mu_cached/sigma_cached
 * from priors when n_obs == 0, so copying is always valid.
 *═══════════════════════════════════════════════════════════════════════════*/

static void sync_storvik_to_rbpf(RBPF_Extended *ext)
{
    if (!ext->storvik_initialized)
        return;
    if (ext->param_mode != RBPF_PARAM_STORVIK)
        return;

    RBPF_KSC *rbpf = ext->rbpf;
    StorvikSoA *soa = &ext->storvik.storvik;
    int total = rbpf->n_particles * rbpf->n_regimes;

    /* Element-by-element copy with type conversion
     * CRITICAL: param_real is double, rbpf_real_t is float
     * memcpy would copy raw double bits into float → garbage/NaN! */
    for (int i = 0; i < total; i++)
    {
        rbpf->particle_mu_vol[i] = (rbpf_real_t)soa->mu_cached[i];
        rbpf->particle_sigma_vol[i] = (rbpf_real_t)soa->sigma_cached[i];
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * TRANSITION MATRIX LEARNING - INTERNAL HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Update transition counts from particle regime changes
 *
 * Called after each step. Uses exponential forgetting to allow adaptation.
 */
static void update_transition_counts(RBPF_Extended *ext)
{
    if (!ext->trans_learn_enabled)
        return;

    RBPF_KSC *rbpf = ext->rbpf;
    int n = rbpf->n_particles;
    int nr = rbpf->n_regimes;
    double forget = ext->trans_forgetting;

    /* Decay old counts (exponential moving average behavior) */
    for (int i = 0; i < nr; i++)
    {
        for (int j = 0; j < nr; j++)
        {
            ext->trans_counts[i][j] *= forget;
        }
    }

    /* Accumulate new transitions from particles (equally weighted) */
    double weight_share = 1.0 / n;

    for (int k = 0; k < n; k++)
    {
        int current_r = rbpf->regime[k];
        int prev_r = ext->prev_regime[k];

        /* Bounds check */
        if (prev_r >= 0 && prev_r < nr && current_r >= 0 && current_r < nr)
        {
            ext->trans_counts[prev_r][current_r] += weight_share;
        }
    }
}

/**
 * Rebuild the transition LUT from learned counts
 *
 * Converts counts to probabilities using Dirichlet-Multinomial posterior:
 *   P_ij = (N_ij + prior) / sum_k(N_ik + prior)
 */
static void rebuild_transition_lut(RBPF_Extended *ext)
{
    if (!ext->trans_learn_enabled)
        return;

    RBPF_KSC *rbpf = ext->rbpf;
    int nr = rbpf->n_regimes;
    rbpf_real_t flat_matrix[RBPF_MAX_REGIMES * RBPF_MAX_REGIMES];

    double prior_diag = ext->trans_prior_diag;
    double prior_off = ext->trans_prior_off;

    for (int i = 0; i < nr; i++)
    {
        double row_sum = 0.0;

        /* Sum row including priors */
        for (int j = 0; j < nr; j++)
        {
            double prior = (i == j) ? prior_diag : prior_off;
            row_sum += ext->trans_counts[i][j] + prior;
        }

        /* Normalize to probabilities */
        for (int j = 0; j < nr; j++)
        {
            double prior = (i == j) ? prior_diag : prior_off;
            double count = ext->trans_counts[i][j] + prior;
            flat_matrix[i * nr + j] = (rbpf_real_t)(count / row_sum);
        }
    }

    /* Push to core filter */
    rbpf_ksc_build_transition_lut(rbpf, flat_matrix);
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN UPDATE
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_step(RBPF_Extended *ext, rbpf_real_t obs, RBPF_KSC_Output *output)
{
    if (!ext || !ext->rbpf)
        return;

    RBPF_KSC *rbpf = ext->rbpf;
    int n = rbpf->n_particles;

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 0: Signal structural break if flagged
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->structural_break_signaled && ext->storvik_initialized)
    {
        param_learn_signal_structural_break(&ext->storvik);
        ext->structural_break_signaled = 0;
    }

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 1: Run RBPF-KSC update (may resample internally!)
     *═══════════════════════════════════════════════════════════════════════*/
    rbpf_ksc_step(rbpf, obs, output);

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 2: Update Storvik with CORRECT ORDER
     *
     * CRITICAL: If RBPF resampled, rbpf->mu is now permuted but Storvik
     * stats are not. We must align Storvik stats BEFORE updating them.
     *
     * Order:
     *   1. If resampled: permute Storvik stats to match new particle order
     *   2. Extract particle info (with lineage fix for ell_lag)
     *   3. Update Storvik stats (now aligned)
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->storvik_initialized)
    {
        /* STEP 2a: Align Storvik stats to match permuted RBPF state */
        if (output->resampled)
        {
            param_learn_apply_resampling(&ext->storvik, rbpf->indices, n);
        }

        /* STEP 2b: Extract particle info (pass resampled for lineage fix) */
        extract_particle_info(ext, output->resampled);

        /* STEP 2c: Update Storvik stats (now everything is aligned) */
        param_learn_update(&ext->storvik, ext->particle_info, n);
    }

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 3: Update transition matrix learning (if enabled)
     *
     * Must run BEFORE prev_regime is updated, since we need:
     *   prev_regime[k] = regime before this step
     *   rbpf->regime[k] = regime after this step
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->trans_learn_enabled)
    {
        update_transition_counts(ext);

        /* Rebuild LUT periodically or on structural break */
        ext->trans_ticks_since_update++;
        if (ext->trans_ticks_since_update >= ext->trans_update_interval ||
            ext->structural_break_signaled)
        {
            rebuild_transition_lut(ext);
            ext->trans_ticks_since_update = 0;
        }
    }

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 4: Update lag buffers for next tick
     *═══════════════════════════════════════════════════════════════════════*/
    for (int i = 0; i < n; i++)
    {
        ext->ell_lag_buffer[i] = rbpf->mu[i];  /* Current becomes lag */
        ext->prev_regime[i] = rbpf->regime[i]; /* Track regime changes */
    }

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 5: Sync learned params back to RBPF (if Storvik mode)
     *═══════════════════════════════════════════════════════════════════════*/
    sync_storvik_to_rbpf(ext);

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 6: Populate learned params in output
     *═══════════════════════════════════════════════════════════════════════*/
    for (int r = 0; r < rbpf->n_regimes; r++)
    {
        rbpf_ext_get_learned_params(ext, r,
                                    &output->learned_mu_vol[r],
                                    &output->learned_sigma_vol[r]);
    }
}

void rbpf_ext_step_apf(RBPF_Extended *ext, rbpf_real_t obs_current,
                       rbpf_real_t obs_next, RBPF_KSC_Output *output)
{
    if (!ext || !ext->rbpf)
        return;

    RBPF_KSC *rbpf = ext->rbpf;
    int n = rbpf->n_particles;
    int n_regimes = rbpf->n_regimes;

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 0: Signal structural break if flagged
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->structural_break_signaled && ext->storvik_initialized)
    {
        param_learn_signal_structural_break(&ext->storvik);
        ext->structural_break_signaled = 0;
    }

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 1: Run APF step WITH INDEX OUTPUT
     *
     * CRITICAL: We need the resample indices to apply the SAME resampling
     * to Storvik per-particle arrays. Without this, particle states and
     * their learned parameters become misaligned → NaN/garbage.
     *═══════════════════════════════════════════════════════════════════════*/

    /* Allocate indices on stack (reasonable for n ≤ 2048) */
    int resample_indices[2048];
    int *indices = (n <= 2048) ? resample_indices : (int *)malloc(n * sizeof(int));

    /* Use indexed APF step - returns resample indices */
    rbpf_ksc_step_apf_indexed(rbpf, obs_current, obs_next, output, indices);

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 2: Update Storvik with CORRECT ORDER
     *
     * CRITICAL: APF already resampled RBPF arrays. We must:
     *   1. Align Storvik stats to match new particle order FIRST
     *   2. Extract particle info with lineage fix
     *   3. Update stats (now aligned)
     *   4. Also resample RBPF per-particle param arrays
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->storvik_initialized)
    {
        /* STEP 2a: Align Storvik stats BEFORE update */
        if (output->resampled)
        {
            param_learn_apply_resampling(&ext->storvik, indices, n);
        }

        /* STEP 2b: Extract particle info with lineage fix */
        extract_particle_info(ext, output->resampled);

        /* STEP 2c: Update Storvik sufficient statistics */
        param_learn_update(&ext->storvik, ext->particle_info, n);

        /*═══════════════════════════════════════════════════════════════════
         * STEP 2d: Also resample RBPF per-particle param arrays
         *
         * CRITICAL FIX: rbpf->particle_mu_vol and particle_sigma_vol must
         * be resampled with the same indices as mu/var/regime!
         *═══════════════════════════════════════════════════════════════════*/
        if (output->resampled && rbpf->particle_mu_vol && rbpf->particle_sigma_vol)
        {
            /* Use scratch space for double buffering */
            rbpf_real_t *mu_vol_new = (rbpf_real_t *)malloc(n * n_regimes * sizeof(rbpf_real_t));
            rbpf_real_t *sigma_vol_new = (rbpf_real_t *)malloc(n * n_regimes * sizeof(rbpf_real_t));

            if (mu_vol_new && sigma_vol_new)
            {
                for (int i = 0; i < n; i++)
                {
                    int src = indices[i];
                    for (int r = 0; r < n_regimes; r++)
                    {
                        int dst_idx = i * n_regimes + r;
                        int src_idx = src * n_regimes + r;
                        mu_vol_new[dst_idx] = rbpf->particle_mu_vol[src_idx];
                        sigma_vol_new[dst_idx] = rbpf->particle_sigma_vol[src_idx];
                    }
                }

                /* Copy back */
                memcpy(rbpf->particle_mu_vol, mu_vol_new, n * n_regimes * sizeof(rbpf_real_t));
                memcpy(rbpf->particle_sigma_vol, sigma_vol_new, n * n_regimes * sizeof(rbpf_real_t));
            }

            free(mu_vol_new);
            free(sigma_vol_new);
        }
    }

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 3: Update transition matrix learning (if enabled)
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->trans_learn_enabled)
    {
        update_transition_counts(ext);

        ext->trans_ticks_since_update++;
        if (ext->trans_ticks_since_update >= ext->trans_update_interval ||
            ext->structural_break_signaled)
        {
            rebuild_transition_lut(ext);
            ext->trans_ticks_since_update = 0;
        }
    }

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 4: Update lag buffers for next tick
     *═══════════════════════════════════════════════════════════════════════*/
    for (int i = 0; i < n; i++)
    {
        ext->ell_lag_buffer[i] = rbpf->mu[i];
        ext->prev_regime[i] = rbpf->regime[i];
    }

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 5: Sync learned params back to RBPF (if Storvik mode)
     *═══════════════════════════════════════════════════════════════════════*/
    sync_storvik_to_rbpf(ext);

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 6: Populate learned params in output
     *═══════════════════════════════════════════════════════════════════════*/
    for (int r = 0; r < n_regimes; r++)
    {
        rbpf_ext_get_learned_params(ext, r,
                                    &output->learned_mu_vol[r],
                                    &output->learned_sigma_vol[r]);
    }

    /* Cleanup if we allocated dynamically */
    if (n > 2048)
    {
        free(indices);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * PARAMETER ACCESS
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_get_learned_params(const RBPF_Extended *ext, int regime,
                                 rbpf_real_t *mu_vol, rbpf_real_t *sigma_vol)
{
    if (!ext || regime < 0 || regime >= RBPF_MAX_REGIMES)
    {
        if (mu_vol)
            *mu_vol = RBPF_REAL(-4.6); /* Default: ~1% vol */
        if (sigma_vol)
            *sigma_vol = RBPF_REAL(0.1);
        return;
    }

    switch (ext->param_mode)
    {
    case RBPF_PARAM_STORVIK:
    case RBPF_PARAM_HYBRID:
        /* Get from Storvik (weighted average across particles) */
        if (ext->storvik_initialized)
        {
            RegimeParams params;
            param_learn_get_params(&ext->storvik, 0, regime, &params);
            if (mu_vol)
                *mu_vol = params.mu;
            if (sigma_vol)
                *sigma_vol = params.sigma;
        }
        else
        {
            if (mu_vol)
                *mu_vol = ext->rbpf->params[regime].mu_vol;
            if (sigma_vol)
                *sigma_vol = ext->rbpf->params[regime].sigma_vol;
        }
        break;

    case RBPF_PARAM_LIU_WEST:
        /* Get from Liu-West */
        rbpf_ksc_get_learned_params(ext->rbpf, regime, mu_vol, sigma_vol);
        break;

    default:
        /* Fixed params */
        if (mu_vol)
            *mu_vol = ext->rbpf->params[regime].mu_vol;
        if (sigma_vol)
            *sigma_vol = ext->rbpf->params[regime].sigma_vol;
        break;
    }
}

void rbpf_ext_get_storvik_summary(const RBPF_Extended *ext, int regime,
                                  RegimeParams *summary)
{
    if (!ext || !summary || !ext->storvik_initialized)
    {
        if (summary)
            memset(summary, 0, sizeof(RegimeParams));
        return;
    }

    param_learn_get_params(&ext->storvik, 0, regime, summary);
}

void rbpf_ext_get_learning_stats(const RBPF_Extended *ext,
                                 uint64_t *stat_updates,
                                 uint64_t *samples_drawn,
                                 uint64_t *samples_skipped)
{
    if (!ext || !ext->storvik_initialized)
    {
        if (stat_updates)
            *stat_updates = 0;
        if (samples_drawn)
            *samples_drawn = 0;
        if (samples_skipped)
            *samples_skipped = 0;
        return;
    }

    if (stat_updates)
        *stat_updates = ext->storvik.total_stat_updates;
    if (samples_drawn)
        *samples_drawn = ext->storvik.total_samples_drawn;
    if (samples_skipped)
        *samples_skipped = ext->storvik.samples_skipped_load;
}

/*═══════════════════════════════════════════════════════════════════════════
 * DEBUG
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_print_config(const RBPF_Extended *ext)
{
    if (!ext)
        return;

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         RBPF-KSC Extended Configuration                      ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    const char *mode_str;
    switch (ext->param_mode)
    {
    case RBPF_PARAM_DISABLED:
        mode_str = "DISABLED (fixed params)";
        break;
    case RBPF_PARAM_LIU_WEST:
        mode_str = "LIU-WEST (fast adaptation)";
        break;
    case RBPF_PARAM_STORVIK:
        mode_str = "STORVIK (full Bayesian)";
        break;
    case RBPF_PARAM_HYBRID:
        mode_str = "HYBRID (both)";
        break;
    default:
        mode_str = "UNKNOWN";
        break;
    }

    printf("Parameter Learning: %s\n", mode_str);
    printf("Particles:          %d\n", ext->rbpf->n_particles);
    printf("Regimes:            %d\n", ext->rbpf->n_regimes);

    if (ext->storvik_initialized)
    {
        printf("\nStorvik Sampling Intervals:\n");
        for (int r = 0; r < ext->rbpf->n_regimes; r++)
        {
            printf("  R%d: every %d ticks\n", r, ext->storvik.config.sample_interval[r]);
        }
    }

    printf("\nPer-Regime Parameters:\n");
    printf("  %-8s %10s %10s %10s\n", "Regime", "theta", "mu_vol", "sigma_vol");
    printf("  %-8s %10s %10s %10s\n", "------", "-----", "------", "---------");

    for (int r = 0; r < ext->rbpf->n_regimes; r++)
    {
        const RBPF_RegimeParams *p = &ext->rbpf->params[r];
        printf("  %-8d %10.4f %10.4f %10.4f\n", r, p->theta, p->mu_vol, p->sigma_vol);
    }

    printf("\n");
}

void rbpf_ext_print_storvik_stats(const RBPF_Extended *ext, int regime)
{
    if (!ext || !ext->storvik_initialized)
        return;

    printf("\nStorvik Statistics (Regime %d):\n", regime);
    param_learn_print_regime_stats(&ext->storvik, regime);
}