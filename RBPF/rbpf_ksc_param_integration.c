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

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

RBPF_Extended* rbpf_ext_create(int n_particles, int n_regimes, RBPF_ParamMode mode)
{
    RBPF_Extended *ext = (RBPF_Extended*)calloc(1, sizeof(RBPF_Extended));
    if (!ext) return NULL;
    
    ext->param_mode = mode;
    
    /* Create core RBPF-KSC */
    ext->rbpf = rbpf_ksc_create(n_particles, n_regimes);
    if (!ext->rbpf) {
        free(ext);
        return NULL;
    }
    
    /* Allocate workspace */
    ext->particle_info = (ParticleInfo*)calloc(n_particles, sizeof(ParticleInfo));
    ext->prev_regime = (int*)calloc(n_particles, sizeof(int));
    ext->ell_lag_buffer = (rbpf_real_t*)calloc(n_particles, sizeof(rbpf_real_t));
    
    if (!ext->particle_info || !ext->prev_regime || !ext->ell_lag_buffer) {
        rbpf_ext_destroy(ext);
        return NULL;
    }
    
    /* Initialize Storvik if needed */
    if (mode == RBPF_PARAM_STORVIK || mode == RBPF_PARAM_HYBRID) {
        ParamLearnConfig cfg = param_learn_config_defaults();
        
        /* Default intervals: sleep in calm, wake in crisis */
        cfg.sample_interval[0] = 50;   /* R0: every 50 ticks */
        cfg.sample_interval[1] = 20;   /* R1: every 20 ticks */
        cfg.sample_interval[2] = 5;    /* R2: every 5 ticks */
        cfg.sample_interval[3] = 1;    /* R3: every tick */
        
        cfg.sample_on_regime_change = true;
        cfg.sample_on_structural_break = true;
        cfg.sample_on_resample = true;
        
        if (param_learn_init(&ext->storvik, &cfg, n_particles, n_regimes) != 0) {
            rbpf_ext_destroy(ext);
            return NULL;
        }
        ext->storvik_initialized = 1;
    }
    
    /*═══════════════════════════════════════════════════════════════════════
     * CRITICAL: Configure per-particle parameter mode in core RBPF
     *
     * The RBPF predict step checks liu_west.enabled to decide whether to
     * use particle_mu_vol[] or global params[].theta/mu_vol.
     *
     * In STORVIK mode, we MUST enable this flag so the predict step reads
     * the per-particle arrays that sync_storvik_to_rbpf() populates.
     * Setting shrinkage=1.0 and warmup=0 makes internal Liu-West logic
     * a no-op (we overwrite with Storvik values anyway).
     *═══════════════════════════════════════════════════════════════════════*/
    if (mode == RBPF_PARAM_STORVIK) {
        /* Enable per-particle params in predict, but Storvik controls values */
        rbpf_ksc_enable_liu_west(ext->rbpf, 1.0f, 0);
        /* Disable internal Liu-West update logic (we overwrite anyway) */
        ext->rbpf->liu_west.learn_mu_vol = 0;
        ext->rbpf->liu_west.learn_sigma_vol = 0;
    }
    else if (mode == RBPF_PARAM_LIU_WEST || mode == RBPF_PARAM_HYBRID) {
        /* Standard Liu-West configuration */
        rbpf_ksc_enable_liu_west(ext->rbpf, 0.98f, 100);
    }
    
    return ext;
}

void rbpf_ext_destroy(RBPF_Extended *ext)
{
    if (!ext) return;
    
    if (ext->rbpf) {
        rbpf_ksc_destroy(ext->rbpf);
    }
    
    if (ext->storvik_initialized) {
        param_learn_free(&ext->storvik);
    }
    
    free(ext->particle_info);
    free(ext->prev_regime);
    free(ext->ell_lag_buffer);
    free(ext);
}

void rbpf_ext_init(RBPF_Extended *ext, rbpf_real_t mu0, rbpf_real_t var0)
{
    if (!ext) return;
    
    /* Initialize RBPF state */
    rbpf_ksc_init(ext->rbpf, mu0, var0);
    
    /* Initialize lag buffers */
    int n = ext->rbpf->n_particles;
    for (int i = 0; i < n; i++) {
        ext->ell_lag_buffer[i] = mu0;
        ext->prev_regime[i] = ext->rbpf->regime[i];
    }
    
    /* Initialize Storvik priors to match RBPF params */
    if (ext->storvik_initialized) {
        int nr = ext->rbpf->n_regimes;
        for (int r = 0; r < nr; r++) {
            const RBPF_RegimeParams *p = &ext->rbpf->params[r];
            param_learn_set_prior(&ext->storvik, r, p->mu_vol, p->theta, p->sigma_vol);
        }
        param_learn_broadcast_priors(&ext->storvik);
    }
    
    ext->structural_break_signaled = 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_set_regime_params(RBPF_Extended *ext, int regime,
                                 rbpf_real_t theta, rbpf_real_t mu_vol, rbpf_real_t sigma_vol)
{
    if (!ext || regime < 0 || regime >= RBPF_MAX_REGIMES) return;
    
    /* Set in RBPF */
    rbpf_ksc_set_regime_params(ext->rbpf, regime, theta, mu_vol, sigma_vol);
    
    /* Set in Storvik */
    if (ext->storvik_initialized) {
        param_learn_set_prior(&ext->storvik, regime, mu_vol, theta, sigma_vol);
    }
}

void rbpf_ext_build_transition_lut(RBPF_Extended *ext, const rbpf_real_t *trans_matrix)
{
    if (!ext) return;
    rbpf_ksc_build_transition_lut(ext->rbpf, trans_matrix);
}

void rbpf_ext_set_storvik_interval(RBPF_Extended *ext, int regime, int interval)
{
    if (!ext || !ext->storvik_initialized) return;
    if (regime < 0 || regime >= PARAM_LEARN_MAX_REGIMES) return;
    
    ext->storvik.config.sample_interval[regime] = interval;
}

void rbpf_ext_set_hft_mode(RBPF_Extended *ext, int enable)
{
    if (!ext || !ext->storvik_initialized) return;
    
    if (enable) {
        /* HFT: Less frequent sampling in calm markets */
        ext->storvik.config.sample_interval[0] = 100;  /* R0: every 100 ticks */
        ext->storvik.config.sample_interval[1] = 50;   /* R1: every 50 ticks */
        ext->storvik.config.sample_interval[2] = 20;   /* R2: every 20 ticks */
        ext->storvik.config.sample_interval[3] = 5;    /* R3: every 5 ticks */
    } else {
        /* Standard: More frequent sampling */
        ext->storvik.config.sample_interval[0] = 50;
        ext->storvik.config.sample_interval[1] = 20;
        ext->storvik.config.sample_interval[2] = 5;
        ext->storvik.config.sample_interval[3] = 1;
    }
}

void rbpf_ext_signal_structural_break(RBPF_Extended *ext)
{
    if (!ext) return;
    
    ext->structural_break_signaled = 1;
    
    if (ext->storvik_initialized) {
        param_learn_signal_structural_break(&ext->storvik);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * INTERNAL: Extract particle info for Storvik
 *═══════════════════════════════════════════════════════════════════════════*/

static void extract_particle_info(RBPF_Extended *ext)
{
    RBPF_KSC *rbpf = ext->rbpf;
    int n = rbpf->n_particles;
    
    /* Normalize weights for Storvik */
    rbpf_real_t max_lw = rbpf->log_weight[0];
    for (int i = 1; i < n; i++) {
        if (rbpf->log_weight[i] > max_lw) max_lw = rbpf->log_weight[i];
    }
    
    rbpf_real_t sum_w = RBPF_REAL(0.0);
    for (int i = 0; i < n; i++) {
        ext->particle_info[i].weight = rbpf_exp(rbpf->log_weight[i] - max_lw);
        sum_w += ext->particle_info[i].weight;
    }
    
    rbpf_real_t inv_sum = RBPF_REAL(1.0) / (sum_w + RBPF_REAL(1e-30));
    
    for (int i = 0; i < n; i++) {
        ParticleInfo *p = &ext->particle_info[i];
        
        p->regime = rbpf->regime[i];
        p->prev_regime = ext->prev_regime[i];
        p->ell = rbpf->mu[i];                    /* Current log-vol estimate */
        p->ell_lag = ext->ell_lag_buffer[i];     /* Previous log-vol estimate */
        p->weight *= inv_sum;                     /* Normalize */
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * INTERNAL: Push learned params from Storvik to RBPF (Vectorized)
 *
 * Both arrays use [particle * n_regimes + regime] layout and are contiguous.
 * Direct memcpy is safe because Storvik initializes mu_cached/sigma_cached
 * from priors when n_obs == 0, so copying blindly is correct.
 *═══════════════════════════════════════════════════════════════════════════*/

static void sync_storvik_to_rbpf(RBPF_Extended *ext)
{
    if (!ext->storvik_initialized) return;
    if (ext->param_mode != RBPF_PARAM_STORVIK) return;
    
    RBPF_KSC *rbpf = ext->rbpf;
    StorvikSoA *soa = &ext->storvik.storvik;
    size_t total_bytes = (size_t)rbpf->n_particles * rbpf->n_regimes * sizeof(rbpf_real_t);
    
    /* Direct bulk copy - O(1) vectorized vs O(N×R) scalar */
    memcpy(rbpf->particle_mu_vol, soa->mu_cached, total_bytes);
    memcpy(rbpf->particle_sigma_vol, soa->sigma_cached, total_bytes);
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN UPDATE
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_step(RBPF_Extended *ext, rbpf_real_t obs, RBPF_KSC_Output *output)
{
    if (!ext || !ext->rbpf) return;
    
    RBPF_KSC *rbpf = ext->rbpf;
    int n = rbpf->n_particles;
    
    /*═══════════════════════════════════════════════════════════════════════
     * STEP 0: Signal structural break if flagged
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->structural_break_signaled && ext->storvik_initialized) {
        param_learn_signal_structural_break(&ext->storvik);
        ext->structural_break_signaled = 0;
    }
    
    /*═══════════════════════════════════════════════════════════════════════
     * STEP 1: Run RBPF-KSC update
     *═══════════════════════════════════════════════════════════════════════*/
    rbpf_ksc_step(rbpf, obs, output);
    
    /*═══════════════════════════════════════════════════════════════════════
     * STEP 2: Extract particle info and update Storvik
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->storvik_initialized) {
        /* Extract particle info from RBPF state */
        extract_particle_info(ext);
        
        /* Update Storvik (stats always, samples conditionally) */
        param_learn_update(&ext->storvik, ext->particle_info, n);
        
        /* If resample occurred, sync Storvik ancestors */
        if (output->resampled) {
            param_learn_apply_resampling(&ext->storvik, rbpf->indices, n);
        }
    }
    
    /*═══════════════════════════════════════════════════════════════════════
     * STEP 3: Update lag buffers for next tick
     *═══════════════════════════════════════════════════════════════════════*/
    for (int i = 0; i < n; i++) {
        ext->ell_lag_buffer[i] = rbpf->mu[i];   /* Current becomes lag */
        ext->prev_regime[i] = rbpf->regime[i];  /* Track regime changes */
    }
    
    /*═══════════════════════════════════════════════════════════════════════
     * STEP 4: Sync learned params back to RBPF (if Storvik mode)
     *═══════════════════════════════════════════════════════════════════════*/
    sync_storvik_to_rbpf(ext);
    
    /*═══════════════════════════════════════════════════════════════════════
     * STEP 5: Populate learned params in output
     *═══════════════════════════════════════════════════════════════════════*/
    for (int r = 0; r < rbpf->n_regimes; r++) {
        rbpf_ext_get_learned_params(ext, r, 
                                     &output->learned_mu_vol[r],
                                     &output->learned_sigma_vol[r]);
    }
}

void rbpf_ext_step_apf(RBPF_Extended *ext, rbpf_real_t obs_current, 
                       rbpf_real_t obs_next, RBPF_KSC_Output *output)
{
    if (!ext || !ext->rbpf) return;
    
    RBPF_KSC *rbpf = ext->rbpf;
    int n = rbpf->n_particles;
    
    /* Signal structural break if flagged */
    if (ext->structural_break_signaled && ext->storvik_initialized) {
        param_learn_signal_structural_break(&ext->storvik);
        ext->structural_break_signaled = 0;
    }
    
    /* Run APF step */
    rbpf_ksc_step_apf(rbpf, obs_current, obs_next, output);
    
    /* Update Storvik */
    if (ext->storvik_initialized) {
        extract_particle_info(ext);
        param_learn_update(&ext->storvik, ext->particle_info, n);
        
        if (output->resampled) {
            param_learn_apply_resampling(&ext->storvik, rbpf->indices, n);
        }
    }
    
    /* Update lag buffers */
    for (int i = 0; i < n; i++) {
        ext->ell_lag_buffer[i] = rbpf->mu[i];
        ext->prev_regime[i] = rbpf->regime[i];
    }
    
    /* Sync and populate output */
    sync_storvik_to_rbpf(ext);
    
    for (int r = 0; r < rbpf->n_regimes; r++) {
        rbpf_ext_get_learned_params(ext, r, 
                                     &output->learned_mu_vol[r],
                                     &output->learned_sigma_vol[r]);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * PARAMETER ACCESS
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_get_learned_params(const RBPF_Extended *ext, int regime,
                                  rbpf_real_t *mu_vol, rbpf_real_t *sigma_vol)
{
    if (!ext || regime < 0 || regime >= RBPF_MAX_REGIMES) {
        if (mu_vol) *mu_vol = RBPF_REAL(-4.6);  /* Default: ~1% vol */
        if (sigma_vol) *sigma_vol = RBPF_REAL(0.1);
        return;
    }
    
    switch (ext->param_mode) {
        case RBPF_PARAM_STORVIK:
        case RBPF_PARAM_HYBRID:
            /* Get from Storvik (weighted average across particles) */
            if (ext->storvik_initialized) {
                RegimeParams params;
                param_learn_get_params(&ext->storvik, 0, regime, &params);
                if (mu_vol) *mu_vol = params.mu;
                if (sigma_vol) *sigma_vol = params.sigma;
            } else {
                if (mu_vol) *mu_vol = ext->rbpf->params[regime].mu_vol;
                if (sigma_vol) *sigma_vol = ext->rbpf->params[regime].sigma_vol;
            }
            break;
            
        case RBPF_PARAM_LIU_WEST:
            /* Get from Liu-West */
            rbpf_ksc_get_learned_params(ext->rbpf, regime, mu_vol, sigma_vol);
            break;
            
        default:
            /* Fixed params */
            if (mu_vol) *mu_vol = ext->rbpf->params[regime].mu_vol;
            if (sigma_vol) *sigma_vol = ext->rbpf->params[regime].sigma_vol;
            break;
    }
}

void rbpf_ext_get_storvik_summary(const RBPF_Extended *ext, int regime,
                                   RegimeParams *summary)
{
    if (!ext || !summary || !ext->storvik_initialized) {
        if (summary) memset(summary, 0, sizeof(RegimeParams));
        return;
    }
    
    param_learn_get_params(&ext->storvik, 0, regime, summary);
}

void rbpf_ext_get_learning_stats(const RBPF_Extended *ext,
                                  uint64_t *stat_updates,
                                  uint64_t *samples_drawn,
                                  uint64_t *samples_skipped)
{
    if (!ext || !ext->storvik_initialized) {
        if (stat_updates) *stat_updates = 0;
        if (samples_drawn) *samples_drawn = 0;
        if (samples_skipped) *samples_skipped = 0;
        return;
    }
    
    if (stat_updates) *stat_updates = ext->storvik.total_stat_updates;
    if (samples_drawn) *samples_drawn = ext->storvik.total_samples_drawn;
    if (samples_skipped) *samples_skipped = ext->storvik.samples_skipped_load;
}

/*═══════════════════════════════════════════════════════════════════════════
 * DEBUG
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_print_config(const RBPF_Extended *ext)
{
    if (!ext) return;
    
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         RBPF-KSC Extended Configuration                      ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    const char *mode_str;
    switch (ext->param_mode) {
        case RBPF_PARAM_DISABLED: mode_str = "DISABLED (fixed params)"; break;
        case RBPF_PARAM_LIU_WEST: mode_str = "LIU-WEST (fast adaptation)"; break;
        case RBPF_PARAM_STORVIK:  mode_str = "STORVIK (full Bayesian)"; break;
        case RBPF_PARAM_HYBRID:   mode_str = "HYBRID (both)"; break;
        default: mode_str = "UNKNOWN"; break;
    }
    
    printf("Parameter Learning: %s\n", mode_str);
    printf("Particles:          %d\n", ext->rbpf->n_particles);
    printf("Regimes:            %d\n", ext->rbpf->n_regimes);
    
    if (ext->storvik_initialized) {
        printf("\nStorvik Sampling Intervals:\n");
        for (int r = 0; r < ext->rbpf->n_regimes; r++) {
            printf("  R%d: every %d ticks\n", r, ext->storvik.config.sample_interval[r]);
        }
    }
    
    printf("\nPer-Regime Parameters:\n");
    printf("  %-8s %10s %10s %10s\n", "Regime", "theta", "mu_vol", "sigma_vol");
    printf("  %-8s %10s %10s %10s\n", "------", "-----", "------", "---------");
    
    for (int r = 0; r < ext->rbpf->n_regimes; r++) {
        const RBPF_RegimeParams *p = &ext->rbpf->params[r];
        printf("  %-8d %10.4f %10.4f %10.4f\n", r, p->theta, p->mu_vol, p->sigma_vol);
    }
    
    printf("\n");
}

void rbpf_ext_print_storvik_stats(const RBPF_Extended *ext, int regime)
{
    if (!ext || !ext->storvik_initialized) return;
    
    printf("\nStorvik Statistics (Regime %d):\n", regime);
    param_learn_print_regime_stats(&ext->storvik, regime);
}
