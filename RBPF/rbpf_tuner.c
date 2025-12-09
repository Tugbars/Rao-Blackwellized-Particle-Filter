/**
 * @file rbpf_tuner.c
 * @brief RBPF Parameter Auto-Tuner Implementation
 */

#include "rbpf_tuner.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
#include <sys/time.h>
static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * DEFAULT CONFIGURATIONS
 *═══════════════════════════════════════════════════════════════════════════*/

TunerConfig tuner_config_defaults(void)
{
    TunerConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    
    /* Regime separation */
    cfg.mu_vol_base = -5.0f;
    cfg.mu_vol_gap_min = 1.0f;
    cfg.mu_vol_gap_max = 1.8f;
    cfg.mu_vol_gap_steps = 5;
    
    /* Sigma_vol derived from gap */
    cfg.search_sigma_vol = 0;
    cfg.sigma_vol_base = 0.08f;
    cfg.sigma_vol_scale_min = 1.1f;
    cfg.sigma_vol_scale_max = 1.3f;
    cfg.sigma_vol_steps = 3;
    
    /* Transition matrix */
    cfg.self_trans_min = 0.96f;
    cfg.self_trans_max = 0.995f;
    cfg.self_trans_steps = 5;
    cfg.adjacent_only = 1;
    
    /* Hysteresis */
    cfg.hold_min = 5;
    cfg.hold_max = 20;
    cfg.hold_steps = 4;
    
    cfg.prob_thresh_min = 0.65f;
    cfg.prob_thresh_max = 0.90f;
    cfg.prob_thresh_steps = 4;
    
    /* Theta (fixed by default) */
    cfg.search_theta = 0;
    cfg.theta_base = 0.03f;
    cfg.theta_increment = 0.02f;
    
    /* Evaluation */
    cfg.n_particles = 512;
    cfg.n_regimes = 4;
    cfg.n_eval_runs = 2;
    cfg.warmup_ticks = 200;
    cfg.base_seed = 42;
    
    /* Objective weights */
    cfg.weight_accuracy = 1.0f;
    cfg.weight_middle_regimes = 1.5f;  /* Bonus for R1, R2 */
    cfg.weight_stability = 0.05f;
    cfg.weight_vol_rmse = 0.0f;        /* Disabled by default */
    
    /* Storvik */
    cfg.use_storvik = 1;
    cfg.storvik_prior_strength = 10.0f;
    
    /* Output */
    cfg.verbose = 1;
    cfg.print_every_n = 20;
    
    return cfg;
}

TunerConfig tuner_config_fast(void)
{
    TunerConfig cfg = tuner_config_defaults();
    
    cfg.mu_vol_gap_steps = 3;
    cfg.self_trans_steps = 3;
    cfg.hold_steps = 3;
    cfg.prob_thresh_steps = 3;
    cfg.n_eval_runs = 1;
    
    return cfg;
}

TunerConfig tuner_config_detailed(void)
{
    TunerConfig cfg = tuner_config_defaults();
    
    cfg.mu_vol_gap_steps = 8;
    cfg.self_trans_steps = 8;
    cfg.hold_steps = 6;
    cfg.prob_thresh_steps = 6;
    cfg.n_eval_runs = 3;
    
    return cfg;
}

TunerConfig tuner_config_focus_middle(void)
{
    TunerConfig cfg = tuner_config_defaults();
    
    cfg.weight_middle_regimes = 3.0f;  /* Heavy bonus for R1, R2 */
    cfg.mu_vol_gap_min = 1.2f;         /* Ensure good separation */
    cfg.mu_vol_gap_max = 2.0f;
    
    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

int tuner_init(RBPFTuner *tuner, const TunerConfig *cfg,
               const float *returns, const int *true_regimes,
               const float *true_vol, int n_ticks)
{
    if (!tuner || !returns || !true_regimes || n_ticks < 100)
        return -1;
    
    memset(tuner, 0, sizeof(*tuner));
    tuner->config = cfg ? *cfg : tuner_config_defaults();
    
    /* Allocate and copy test data */
    tuner->returns = (float*)malloc(n_ticks * sizeof(float));
    tuner->true_regimes = (int*)malloc(n_ticks * sizeof(int));
    if (!tuner->returns || !tuner->true_regimes) {
        tuner_free(tuner);
        return -1;
    }
    memcpy(tuner->returns, returns, n_ticks * sizeof(float));
    memcpy(tuner->true_regimes, true_regimes, n_ticks * sizeof(int));
    
    if (true_vol) {
        tuner->true_vol = (float*)malloc(n_ticks * sizeof(float));
        if (!tuner->true_vol) {
            tuner_free(tuner);
            return -1;
        }
        memcpy(tuner->true_vol, true_vol, n_ticks * sizeof(float));
    }
    
    tuner->n_ticks = n_ticks;
    
    /* Compute total evaluations */
    const TunerConfig *c = &tuner->config;
    tuner->total_evals = c->mu_vol_gap_steps * c->self_trans_steps *
                         c->hold_steps * c->prob_thresh_steps;
    
    /* Initialize best with worst possible score */
    tuner->best.objective = -FLT_MAX;
    
    tuner->initialized = 1;
    
    return 0;
}

void tuner_free(RBPFTuner *tuner)
{
    if (!tuner) return;
    
    free(tuner->returns);
    free(tuner->true_regimes);
    free(tuner->true_vol);
    
    memset(tuner, 0, sizeof(*tuner));
}

/*═══════════════════════════════════════════════════════════════════════════
 * TRANSITION MATRIX BUILDER
 *═══════════════════════════════════════════════════════════════════════════*/

static void build_transition_matrix(rbpf_real_t *trans, int n_regimes, 
                                    float self_prob, int adjacent_only)
{
    float off_diag_total = 1.0f - self_prob;
    
    for (int i = 0; i < n_regimes; i++) {
        for (int j = 0; j < n_regimes; j++) {
            int idx = i * n_regimes + j;
            
            if (i == j) {
                trans[idx] = (rbpf_real_t)self_prob;
            } else if (adjacent_only) {
                /* Only allow transitions to adjacent regimes */
                if (abs(i - j) == 1) {
                    if (i == 0 || i == n_regimes - 1) {
                        /* Edge regime: all off-diagonal to single neighbor */
                        trans[idx] = (rbpf_real_t)off_diag_total;
                    } else {
                        /* Middle regime: split between two neighbors */
                        trans[idx] = (rbpf_real_t)(off_diag_total / 2.0f);
                    }
                } else {
                    trans[idx] = 0.0f;
                }
            } else {
                /* Allow all transitions, uniform distribution */
                trans[idx] = (rbpf_real_t)(off_diag_total / (n_regimes - 1));
            }
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * SINGLE EVALUATION
 *═══════════════════════════════════════════════════════════════════════════*/

float tuner_evaluate(RBPFTuner *tuner, TunerResult *result)
{
    if (!tuner || !result || !tuner->initialized)
        return -FLT_MAX;
    
    const TunerConfig *cfg = &tuner->config;
    const int n_regimes = cfg->n_regimes;
    double t_start = get_time_ms();
    
    /* Create RBPF */
    RBPF_KSC *rbpf = rbpf_ksc_create(cfg->n_particles, n_regimes);
    if (!rbpf) return -FLT_MAX;
    
    /* Create Storvik learner if enabled */
    ParamLearner *learner = NULL;
    if (cfg->use_storvik) {
        learner = (ParamLearner*)malloc(sizeof(ParamLearner));
        if (learner) {
            ParamLearnConfig pl_cfg = param_learn_config_defaults();
            pl_cfg.prior_strength = cfg->storvik_prior_strength;
            if (param_learn_init(learner, &pl_cfg, cfg->n_particles, n_regimes) < 0) {
                free(learner);
                learner = NULL;
            }
        }
    }
    
    /* Apply parameters to RBPF */
    for (int r = 0; r < n_regimes; r++) {
        rbpf_ksc_set_regime_params(rbpf, r,
            (rbpf_real_t)result->theta[r],
            (rbpf_real_t)result->mu_vol[r],
            (rbpf_real_t)result->sigma_vol[r]);
    }
    
    /* Build transition matrix */
    rbpf_real_t trans[TUNER_MAX_REGIMES * TUNER_MAX_REGIMES];
    build_transition_matrix(trans, n_regimes, result->self_trans, cfg->adjacent_only);
    rbpf_ksc_build_transition_lut(rbpf, trans);
    
    /* Set hysteresis */
    rbpf_ksc_set_regime_smoothing(rbpf, 
        result->hold_threshold,
        (rbpf_real_t)result->prob_threshold);
    
    /* Apply to Storvik if enabled */
    if (learner) {
        for (int r = 0; r < n_regimes; r++) {
            float phi = 1.0f - result->theta[r];
            param_learn_set_prior(learner, r,
                result->mu_vol[r],
                phi,
                result->sigma_vol[r]);
        }
        param_learn_broadcast_priors(learner);
        
        /* Enable Storvik mode in RBPF */
        rbpf->use_learned_params = 1;
    }
    
    /* Initialize RBPF */
    rbpf_ksc_init(rbpf, (rbpf_real_t)result->mu_vol[0], 0.01f);
    
    /* Run evaluation */
    int correct[TUNER_MAX_REGIMES] = {0};
    int total[TUNER_MAX_REGIMES] = {0};
    int confusion[TUNER_MAX_REGIMES][TUNER_MAX_REGIMES] = {{0}};
    int n_switches = 0;
    int prev_regime = -1;
    double vol_sum_sq_error = 0.0;
    double vol_sum_bias = 0.0;
    int vol_count = 0;
    
    RBPF_KSC_Output out;
    ParticleInfo *pinfo = NULL;
    
    if (learner) {
        pinfo = (ParticleInfo*)malloc(cfg->n_particles * sizeof(ParticleInfo));
    }
    
    for (int t = 0; t < tuner->n_ticks; t++) {
        /* RBPF step */
        rbpf_ksc_step(rbpf, (rbpf_real_t)tuner->returns[t], &out);
        
        /* Storvik update if enabled */
        if (learner && pinfo) {
            /* Build particle info from RBPF state */
            for (int i = 0; i < cfg->n_particles; i++) {
                pinfo[i].regime = rbpf->regime[i];
                pinfo[i].prev_regime = rbpf->regime[i];  /* Simplified */
                pinfo[i].ell = rbpf->mu[i];
                pinfo[i].ell_lag = rbpf->mu[i];  /* Simplified - would need history */
                pinfo[i].weight = 1.0f / cfg->n_particles;
            }
            param_learn_update(learner, pinfo, cfg->n_particles);
            
            /* Copy learned params to RBPF particle arrays */
            StorvikSoA *soa = param_learn_get_active_soa(learner);
            for (int i = 0; i < cfg->n_particles; i++) {
                for (int r = 0; r < n_regimes; r++) {
                    int idx = i * n_regimes + r;
                    rbpf->particle_mu_vol[idx] = (rbpf_real_t)soa->mu_cached[idx];
                    rbpf->particle_sigma_vol[idx] = (rbpf_real_t)soa->sigma_cached[idx];
                }
            }
        }
        
        /* Measure accuracy after warmup */
        if (t >= cfg->warmup_ticks) {
            int true_r = tuner->true_regimes[t];
            int pred_r = out.smoothed_regime;
            
            if (true_r >= 0 && true_r < n_regimes) {
                total[true_r]++;
                if (pred_r >= 0 && pred_r < n_regimes) {
                    confusion[true_r][pred_r]++;
                }
                
                if (pred_r == true_r) {
                    correct[true_r]++;
                }
                
                if (prev_regime >= 0 && pred_r != prev_regime) {
                    n_switches++;
                }
                prev_regime = pred_r;
            }
            
            /* Volatility error if ground truth available */
            if (tuner->true_vol) {
                float pred_vol = (float)out.vol_mean;
                float true_vol = tuner->true_vol[t];
                float error = pred_vol - true_vol;
                vol_sum_sq_error += error * error;
                vol_sum_bias += error;
                vol_count++;
            }
        }
    }
    
    /* Cleanup */
    free(pinfo);
    if (learner) {
        param_learn_free(learner);
        free(learner);
    }
    rbpf_ksc_destroy(rbpf);
    
    /* Compute metrics */
    int total_correct = 0, total_samples = 0;
    for (int r = 0; r < n_regimes; r++) {
        result->regime_accuracy[r] = total[r] > 0 ? 
            (float)correct[r] / total[r] : 0.0f;
        total_correct += correct[r];
        total_samples += total[r];
        
        for (int j = 0; j < n_regimes; j++) {
            result->confusion[r][j] = total[r] > 0 ?
                (float)confusion[r][j] / total[r] : 0.0f;
        }
    }
    
    result->overall_accuracy = total_samples > 0 ?
        (float)total_correct / total_samples : 0.0f;
    result->n_switches = n_switches;
    result->switch_rate = total_samples > 0 ?
        (float)n_switches / total_samples : 0.0f;
    
    if (vol_count > 0) {
        result->vol_rmse = sqrtf((float)(vol_sum_sq_error / vol_count));
        result->vol_bias = (float)(vol_sum_bias / vol_count);
    } else {
        result->vol_rmse = 0.0f;
        result->vol_bias = 0.0f;
    }
    
    /* Compute objective */
    float obj = cfg->weight_accuracy * result->overall_accuracy;
    
    /* Bonus for middle regimes */
    if (n_regimes >= 4) {
        float middle_acc = (result->regime_accuracy[1] + result->regime_accuracy[2]) / 2.0f;
        obj += cfg->weight_middle_regimes * middle_acc;
    }
    
    /* Penalty for excessive switching */
    obj -= cfg->weight_stability * result->switch_rate * 10.0f;
    
    /* Penalty for vol error */
    if (cfg->weight_vol_rmse > 0 && vol_count > 0) {
        obj -= cfg->weight_vol_rmse * result->vol_rmse;
    }
    
    result->objective = obj;
    result->elapsed_ms = (float)(get_time_ms() - t_start);
    
    return obj;
}

float tuner_evaluate_params(RBPFTuner *tuner,
                            const float *mu_vol,
                            const float *sigma_vol,
                            const float *theta,
                            float self_trans,
                            int hold_threshold,
                            float prob_threshold,
                            TunerResult *result)
{
    if (!result) return -FLT_MAX;
    
    int n_regimes = tuner->config.n_regimes;
    memcpy(result->mu_vol, mu_vol, n_regimes * sizeof(float));
    memcpy(result->sigma_vol, sigma_vol, n_regimes * sizeof(float));
    memcpy(result->theta, theta, n_regimes * sizeof(float));
    result->self_trans = self_trans;
    result->hold_threshold = hold_threshold;
    result->prob_threshold = prob_threshold;
    
    return tuner_evaluate(tuner, result);
}

/*═══════════════════════════════════════════════════════════════════════════
 * GRID SEARCH
 *═══════════════════════════════════════════════════════════════════════════*/

float tuner_grid_search(RBPFTuner *tuner)
{
    if (!tuner || !tuner->initialized)
        return -FLT_MAX;
    
    TunerConfig *cfg = &tuner->config;
    const int n_regimes = cfg->n_regimes;
    TunerResult result;
    
    /* Default sigma_vol and theta (can be refined) */
    float sigma_base[TUNER_MAX_REGIMES];
    float theta_vals[TUNER_MAX_REGIMES];
    
    for (int r = 0; r < n_regimes; r++) {
        sigma_base[r] = cfg->sigma_vol_base * powf(1.15f, (float)r);
        theta_vals[r] = cfg->theta_base + cfg->theta_increment * r;
    }
    
    int eval_count = 0;
    double search_start = get_time_ms();
    
    if (cfg->verbose >= 1) {
        printf("\n");
        printf("================================================================\n");
        printf("  RBPF AUTO-TUNER: Grid Search\n");
        printf("================================================================\n");
        printf("  Test data: %d ticks (%d after warmup)\n", 
               tuner->n_ticks, tuner->n_ticks - cfg->warmup_ticks);
        printf("  Particles: %d, Regimes: %d\n", cfg->n_particles, n_regimes);
        printf("  Total evaluations: %d\n", tuner->total_evals * cfg->n_eval_runs);
        printf("================================================================\n\n");
    }
    
    /* Grid search */
    for (int g = 0; g < cfg->mu_vol_gap_steps; g++) {
        float gap = cfg->mu_vol_gap_min + 
            (cfg->mu_vol_gap_max - cfg->mu_vol_gap_min) * 
            (float)g / (float)(cfg->mu_vol_gap_steps - 1);
        
        /* Set mu_vol with this gap */
        result.mu_vol[0] = cfg->mu_vol_base;
        for (int r = 1; r < n_regimes; r++) {
            result.mu_vol[r] = result.mu_vol[r-1] + gap;
        }
        
        /* Scale sigma_vol based on gap */
        for (int r = 0; r < n_regimes; r++) {
            result.sigma_vol[r] = sigma_base[r] * (0.7f + 0.5f * gap / 1.5f);
            result.theta[r] = theta_vals[r];
        }
        
        for (int s = 0; s < cfg->self_trans_steps; s++) {
            result.self_trans = cfg->self_trans_min +
                (cfg->self_trans_max - cfg->self_trans_min) * 
                (float)s / (float)(cfg->self_trans_steps - 1);
            
            for (int h = 0; h < cfg->hold_steps; h++) {
                result.hold_threshold = cfg->hold_min +
                    (cfg->hold_max - cfg->hold_min) * h / (cfg->hold_steps - 1);
                
                for (int p = 0; p < cfg->prob_thresh_steps; p++) {
                    result.prob_threshold = cfg->prob_thresh_min +
                        (cfg->prob_thresh_max - cfg->prob_thresh_min) * 
                        (float)p / (float)(cfg->prob_thresh_steps - 1);
                    
                    /* Average over multiple runs */
                    float sum_obj = 0.0f;
                    TunerResult avg_result = result;
                    memset(avg_result.regime_accuracy, 0, sizeof(avg_result.regime_accuracy));
                    avg_result.overall_accuracy = 0;
                    avg_result.n_switches = 0;
                    
                    for (int run = 0; run < cfg->n_eval_runs; run++) {
                        TunerResult run_result = result;
                        run_result.eval_id = eval_count * cfg->n_eval_runs + run;
                        
                        float obj = tuner_evaluate(tuner, &run_result);
                        sum_obj += obj;
                        
                        avg_result.overall_accuracy += run_result.overall_accuracy;
                        avg_result.n_switches += run_result.n_switches;
                        for (int r = 0; r < n_regimes; r++) {
                            avg_result.regime_accuracy[r] += run_result.regime_accuracy[r];
                            for (int j = 0; j < n_regimes; j++) {
                                avg_result.confusion[r][j] += run_result.confusion[r][j];
                            }
                        }
                    }
                    
                    /* Average */
                    float inv_runs = 1.0f / cfg->n_eval_runs;
                    avg_result.objective = sum_obj * inv_runs;
                    avg_result.overall_accuracy *= inv_runs;
                    avg_result.n_switches = (int)(avg_result.n_switches * inv_runs);
                    for (int r = 0; r < n_regimes; r++) {
                        avg_result.regime_accuracy[r] *= inv_runs;
                        for (int j = 0; j < n_regimes; j++) {
                            avg_result.confusion[r][j] *= inv_runs;
                        }
                    }
                    
                    /* Update best */
                    if (avg_result.objective > tuner->best.objective) {
                        tuner->best = avg_result;
                        
                        if (cfg->verbose >= 1) {
                            printf("[NEW BEST] obj=%.4f acc=%.1f%% ",
                                   avg_result.objective,
                                   avg_result.overall_accuracy * 100);
                            if (n_regimes >= 4) {
                                printf("R1=%.1f%% R2=%.1f%% ",
                                       avg_result.regime_accuracy[1] * 100,
                                       avg_result.regime_accuracy[2] * 100);
                            }
                            printf("\n");
                            printf("         gap=%.2f self=%.3f hold=%d prob=%.2f\n",
                                   gap, result.self_trans, 
                                   result.hold_threshold, result.prob_threshold);
                        }
                    }
                    
                    eval_count++;
                    tuner->completed_evals = eval_count;
                    
                    /* Progress */
                    if (cfg->verbose >= 1 && eval_count % cfg->print_every_n == 0) {
                        double elapsed = get_time_ms() - search_start;
                        double rate = eval_count / (elapsed / 1000.0);
                        int remaining = tuner->total_evals - eval_count;
                        double eta_sec = remaining / rate;
                        
                        printf("Progress: %d/%d (%.1f%%) - %.1f eval/s - ETA: %.0fs\n",
                               eval_count, tuner->total_evals,
                               100.0f * eval_count / tuner->total_evals,
                               rate, eta_sec);
                    }
                }
            }
        }
    }
    
    tuner->total_elapsed_ms = get_time_ms() - search_start;
    
    if (cfg->verbose >= 1) {
        printf("\n");
        printf("================================================================\n");
        printf("  Search complete: %.1f seconds\n", tuner->total_elapsed_ms / 1000.0);
        printf("================================================================\n");
        tuner_print_result(&tuner->best, n_regimes);
    }
    
    return tuner->best.objective;
}

/*═══════════════════════════════════════════════════════════════════════════
 * RESULTS
 *═══════════════════════════════════════════════════════════════════════════*/

const TunerResult* tuner_get_best(const RBPFTuner *tuner)
{
    if (!tuner) return NULL;
    return &tuner->best;
}

void tuner_apply_to_rbpf(const TunerResult *result, RBPF_KSC *rbpf, int n_regimes)
{
    if (!result || !rbpf) return;
    
    for (int r = 0; r < n_regimes; r++) {
        rbpf_ksc_set_regime_params(rbpf, r,
            (rbpf_real_t)result->theta[r],
            (rbpf_real_t)result->mu_vol[r],
            (rbpf_real_t)result->sigma_vol[r]);
    }
    
    rbpf_real_t trans[TUNER_MAX_REGIMES * TUNER_MAX_REGIMES];
    build_transition_matrix(trans, n_regimes, result->self_trans, 1);
    rbpf_ksc_build_transition_lut(rbpf, trans);
    
    rbpf_ksc_set_regime_smoothing(rbpf,
        result->hold_threshold,
        (rbpf_real_t)result->prob_threshold);
}

void tuner_apply_to_storvik(const TunerResult *result, ParamLearner *learner, int n_regimes)
{
    if (!result || !learner) return;
    
    for (int r = 0; r < n_regimes; r++) {
        float phi = 1.0f - result->theta[r];
        param_learn_set_prior(learner, r,
            result->mu_vol[r],
            phi,
            result->sigma_vol[r]);
    }
    
    param_learn_broadcast_priors(learner);
}

void tuner_generate_code(const TunerResult *result, int n_regimes, 
                         char *buffer, int buffer_size)
{
    if (!result || !buffer || buffer_size < 512) return;
    
    int pos = 0;
    pos += snprintf(buffer + pos, buffer_size - pos,
        "/* Auto-generated RBPF parameters */\n"
        "/* Accuracy: %.1f%% | Objective: %.4f */\n\n",
        result->overall_accuracy * 100, result->objective);
    
    pos += snprintf(buffer + pos, buffer_size - pos,
        "/* Regime parameters */\n");
    for (int r = 0; r < n_regimes; r++) {
        pos += snprintf(buffer + pos, buffer_size - pos,
            "rbpf_ksc_set_regime_params(rbpf, %d, %.4ff, %.4ff, %.4ff);\n",
            r, result->theta[r], result->mu_vol[r], result->sigma_vol[r]);
    }
    
    pos += snprintf(buffer + pos, buffer_size - pos,
        "\n/* Transition matrix (self=%.3f, adjacent-only) */\n"
        "rbpf_real_t trans[%d] = {\n",
        result->self_trans, n_regimes * n_regimes);
    
    float off_diag = (1.0f - result->self_trans) / 2.0f;
    for (int i = 0; i < n_regimes; i++) {
        pos += snprintf(buffer + pos, buffer_size - pos, "    ");
        for (int j = 0; j < n_regimes; j++) {
            float val;
            if (i == j) val = result->self_trans;
            else if (abs(i-j) == 1) {
                if (i == 0 || i == n_regimes - 1) val = 1.0f - result->self_trans;
                else val = off_diag;
            } else val = 0.0f;
            
            pos += snprintf(buffer + pos, buffer_size - pos, "%.4ff, ", val);
        }
        pos += snprintf(buffer + pos, buffer_size - pos, "\n");
    }
    pos += snprintf(buffer + pos, buffer_size - pos, "};\n");
    pos += snprintf(buffer + pos, buffer_size - pos,
        "rbpf_ksc_build_transition_lut(rbpf, trans);\n\n");
    
    pos += snprintf(buffer + pos, buffer_size - pos,
        "/* Hysteresis */\n"
        "rbpf_ksc_set_regime_smoothing(rbpf, %d, %.2ff);\n",
        result->hold_threshold, result->prob_threshold);
}

/*═══════════════════════════════════════════════════════════════════════════
 * OUTPUT
 *═══════════════════════════════════════════════════════════════════════════*/

void tuner_print_result(const TunerResult *result, int n_regimes)
{
    if (!result) return;
    
    printf("\n");
    printf("+===========================================================+\n");
    printf("|              TUNER BEST PARAMETERS                        |\n");
    printf("+===========================================================+\n");
    printf("| Objective Score:   %8.4f                               |\n", result->objective);
    printf("| Overall Accuracy:  %7.1f%%                               |\n", result->overall_accuracy * 100);
    printf("+-----------------------------------------------------------+\n");
    printf("| Per-Regime Accuracy:                                      |\n");
    for (int r = 0; r < n_regimes; r++) {
        printf("|   R%d: %5.1f%%                                             |\n",
               r, result->regime_accuracy[r] * 100);
    }
    printf("+-----------------------------------------------------------+\n");
    printf("| Regime Parameters:                                        |\n");
    printf("|   %-6s %8s %8s %8s                       |\n", "Regime", "mu_vol", "sigma", "theta");
    for (int r = 0; r < n_regimes; r++) {
        printf("|   R%-5d %8.3f %8.3f %8.3f                       |\n",
               r, result->mu_vol[r], result->sigma_vol[r], result->theta[r]);
    }
    printf("+-----------------------------------------------------------+\n");
    printf("| Transition & Hysteresis:                                  |\n");
    printf("|   Self-transition: %.4f                                  |\n", result->self_trans);
    printf("|   Hold threshold:  %d ticks                                |\n", result->hold_threshold);
    printf("|   Prob threshold:  %.2f                                    |\n", result->prob_threshold);
    printf("+-----------------------------------------------------------+\n");
    printf("| Stability:                                                |\n");
    printf("|   Regime switches: %d (%.3f/tick)                         |\n", 
           result->n_switches, result->switch_rate);
    printf("+===========================================================+\n");
}

void tuner_print_confusion(const TunerResult *result, int n_regimes)
{
    if (!result) return;
    
    printf("\nConfusion Matrix (rows=true, cols=predicted):\n");
    printf("       ");
    for (int j = 0; j < n_regimes; j++) {
        printf("  R%d   ", j);
    }
    printf("\n");
    
    for (int i = 0; i < n_regimes; i++) {
        printf("  R%d  ", i);
        for (int j = 0; j < n_regimes; j++) {
            printf("%5.1f%% ", result->confusion[i][j] * 100);
        }
        printf("\n");
    }
}

void tuner_print_progress(const RBPFTuner *tuner)
{
    if (!tuner) return;
    
    float pct = 100.0f * tuner->completed_evals / tuner->total_evals;
    printf("Progress: %d/%d (%.1f%%) - Best: %.4f (%.1f%%)\n",
           tuner->completed_evals, tuner->total_evals, pct,
           tuner->best.objective, tuner->best.overall_accuracy * 100);
}

int tuner_export_csv(const RBPFTuner *tuner, const char *filename)
{
    if (!tuner || !filename) return -1;
    
    FILE *f = fopen(filename, "w");
    if (!f) return -1;
    
    const TunerResult *r = &tuner->best;
    int n = tuner->config.n_regimes;
    
    fprintf(f, "parameter,value\n");
    fprintf(f, "objective,%.6f\n", r->objective);
    fprintf(f, "overall_accuracy,%.6f\n", r->overall_accuracy);
    
    for (int i = 0; i < n; i++) {
        fprintf(f, "regime_%d_accuracy,%.6f\n", i, r->regime_accuracy[i]);
        fprintf(f, "mu_vol_%d,%.6f\n", i, r->mu_vol[i]);
        fprintf(f, "sigma_vol_%d,%.6f\n", i, r->sigma_vol[i]);
        fprintf(f, "theta_%d,%.6f\n", i, r->theta[i]);
    }
    
    fprintf(f, "self_trans,%.6f\n", r->self_trans);
    fprintf(f, "hold_threshold,%d\n", r->hold_threshold);
    fprintf(f, "prob_threshold,%.6f\n", r->prob_threshold);
    fprintf(f, "n_switches,%d\n", r->n_switches);
    
    fclose(f);
    return 0;
}
