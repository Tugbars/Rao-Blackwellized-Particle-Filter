/**
 * @file hawkes_intensity.c
 * @brief Hawkes Self-Exciting Point Process Implementation
 */

#include "hawkes_intensity.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

/*═══════════════════════════════════════════════════════════════════════════
 * INTERNAL HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

static inline float maxf(float a, float b) { return a > b ? a : b; }
static inline float minf(float a, float b) { return a < b ? a : b; }
static inline float clampf(float x, float lo, float hi) { 
    return x < lo ? lo : (x > hi ? hi : x); 
}

/**
 * Compute kernel value g(Δt)
 */
static float compute_kernel(const HawkesConfig *cfg, int regime, float dt)
{
    if (dt < 0) return 0.0f;
    
    const HawkesRegimeParams *p = &cfg->regime[regime];
    
    switch (cfg->kernel) {
    case HAWKES_KERNEL_EXPONENTIAL:
        return expf(-p->beta * dt);
        
    case HAWKES_KERNEL_POWER_LAW:
        /* Omori-like: (1 + dt/c)^(-p) where c=1/beta, p=1+beta */
        return powf(1.0f + p->beta * dt, -(1.0f + p->beta));
        
    case HAWKES_KERNEL_SUM_EXP:
        /* Multi-scale: weighted sum of fast and slow decay */
        return p->weight_fast * expf(-p->beta_fast * dt) +
               (1.0f - p->weight_fast) * expf(-p->beta_slow * dt);
               
    default:
        return expf(-p->beta * dt);
    }
}

/**
 * Compute mark impact h(m)
 */
static float compute_mark_impact(const HawkesConfig *cfg, int regime, float mark)
{
    const HawkesRegimeParams *p = &cfg->regime[regime];
    
    switch (cfg->mark_type) {
    case HAWKES_MARK_NONE:
        return 1.0f;
        
    case HAWKES_MARK_LINEAR:
        return mark;
        
    case HAWKES_MARK_QUADRATIC:
        return mark * mark;
        
    case HAWKES_MARK_EXPONENTIAL:
        return expf(p->gamma * mark) - 1.0f;
        
    default:
        return 1.0f;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * DEFAULT CONFIGURATIONS
 *═══════════════════════════════════════════════════════════════════════════*/

HawkesConfig hawkes_config_defaults(void)
{
    HawkesConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    
    cfg.n_regimes = 4;
    cfg.kernel = HAWKES_KERNEL_EXPONENTIAL;
    cfg.mark_type = HAWKES_MARK_LINEAR;
    
    /* Regime 0: Calm - low baseline, weak excitation, fast decay */
    cfg.regime[0].mu = 0.02f;        /* ~2% chance of event per tick */
    cfg.regime[0].alpha = 0.3f;      /* Weak excitation */
    cfg.regime[0].beta = 0.15f;      /* Half-life ≈ 4.6 ticks */
    cfg.regime[0].threshold = 0.02f; /* 2% return triggers event */
    cfg.regime[0].gamma = 1.0f;
    
    /* Regime 1: Mild - moderate baseline, moderate excitation */
    cfg.regime[1].mu = 0.05f;
    cfg.regime[1].alpha = 0.5f;
    cfg.regime[1].beta = 0.12f;      /* Half-life ≈ 5.8 ticks */
    cfg.regime[1].threshold = 0.03f;
    cfg.regime[1].gamma = 1.0f;
    
    /* Regime 2: Elevated - higher baseline, stronger excitation */
    cfg.regime[2].mu = 0.10f;
    cfg.regime[2].alpha = 0.7f;
    cfg.regime[2].beta = 0.08f;      /* Half-life ≈ 8.7 ticks */
    cfg.regime[2].threshold = 0.05f;
    cfg.regime[2].gamma = 1.2f;
    
    /* Regime 3: Crisis - high baseline, strong excitation, slow decay */
    cfg.regime[3].mu = 0.20f;
    cfg.regime[3].alpha = 0.85f;     /* Near critical (α/β < 1 required) */
    cfg.regime[3].beta = 0.05f;      /* Half-life ≈ 13.9 ticks */
    cfg.regime[3].threshold = 0.08f;
    cfg.regime[3].gamma = 1.5f;
    
    /* Transition modification */
    cfg.modify_transitions = 1;
    cfg.intensity_threshold = 0.3f;
    cfg.max_transition_boost = 0.15f;
    
    /* Numerical limits */
    cfg.min_intensity = 0.001f;
    cfg.max_intensity = 5.0f;
    
    return cfg;
}

HawkesConfig hawkes_config_crisis_sensitive(void)
{
    HawkesConfig cfg = hawkes_config_defaults();
    
    /* Lower thresholds to trigger events more easily */
    cfg.regime[0].threshold = 0.015f;
    cfg.regime[1].threshold = 0.02f;
    cfg.regime[2].threshold = 0.03f;
    cfg.regime[3].threshold = 0.05f;
    
    /* Stronger excitation in crisis */
    cfg.regime[3].alpha = 0.90f;
    cfg.regime[3].beta = 0.04f;
    
    /* More aggressive transition modification */
    cfg.intensity_threshold = 0.2f;
    cfg.max_transition_boost = 0.20f;
    
    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

int hawkes_init(HawkesState *state, const HawkesConfig *config)
{
    if (!state) return -1;
    
    memset(state, 0, sizeof(*state));
    state->config = config ? *config : hawkes_config_defaults();
    
    /* Validate branching ratios */
    for (int r = 0; r < state->config.n_regimes; r++) {
        float ratio = hawkes_branching_ratio(state, r);
        if (ratio >= 1.0f) {
            fprintf(stderr, "WARNING: Hawkes regime %d has branching ratio %.2f >= 1 "
                           "(non-stationary). Clamping alpha.\n", r, ratio);
            state->config.regime[r].alpha = state->config.regime[r].beta * 0.95f;
        }
    }
    
    /* Initialize state */
    state->current_regime = 0;
    state->intensity = state->config.regime[0].mu;
    state->intensity_baseline = state->config.regime[0].mu;
    state->intensity_ema = state->intensity;
    
    return 0;
}

void hawkes_reset(HawkesState *state)
{
    if (!state) return;
    
    HawkesConfig cfg = state->config;
    memset(state, 0, sizeof(*state));
    state->config = cfg;
    state->intensity = state->config.regime[0].mu;
    state->intensity_baseline = state->config.regime[0].mu;
    state->intensity_ema = state->intensity;
}

void hawkes_free(HawkesState *state)
{
    if (state) {
        memset(state, 0, sizeof(*state));
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * CORE UPDATE
 *═══════════════════════════════════════════════════════════════════════════*/

float hawkes_update(HawkesState *state, float time, float obs_return, int regime)
{
    if (!state) return 0.0f;
    
    const HawkesConfig *cfg = &state->config;
    regime = clampf(regime, 0, cfg->n_regimes - 1);
    
    const HawkesRegimeParams *rp = &cfg->regime[regime];
    float abs_ret = fabsf(obs_return);
    
    /* Update time */
    float dt = time - state->current_time;
    state->current_time = time;
    state->current_regime = regime;
    
    /* Compute excitation from all past events */
    float excitation = 0.0f;
    
    for (int i = 0; i < state->count; i++) {
        int idx = (state->head - 1 - i + HAWKES_MAX_EVENTS) % HAWKES_MAX_EVENTS;
        HawkesEvent *ev = &state->events[idx];
        
        float event_dt = time - ev->time;
        if (event_dt <= 0) continue;
        
        /* Use regime of event for parameters, or current regime */
        int ev_regime = ev->regime;
        const HawkesRegimeParams *ev_rp = &cfg->regime[ev_regime];
        
        float kernel = compute_kernel(cfg, ev_regime, event_dt);
        float mark_impact = compute_mark_impact(cfg, ev_regime, ev->mark);
        
        excitation += ev_rp->alpha * kernel * mark_impact;
        
        /* Prune old events (kernel effectively zero) */
        if (kernel < 1e-6f && i == state->count - 1) {
            state->count--;
        }
    }
    
    /* Total intensity */
    state->intensity_baseline = rp->mu;
    state->intensity_excited = excitation;
    state->intensity = rp->mu + excitation;
    
    /* Clamp */
    state->intensity = clampf(state->intensity, cfg->min_intensity, cfg->max_intensity);
    
    /* Update EMA */
    float ema_alpha = 0.05f;
    state->intensity_ema = ema_alpha * state->intensity + 
                          (1.0f - ema_alpha) * state->intensity_ema;
    
    /* Check if this observation triggers a new event */
    if (abs_ret > rp->threshold) {
        /* Add event to history */
        HawkesEvent *new_ev = &state->events[state->head];
        new_ev->time = time;
        new_ev->mark = abs_ret;
        new_ev->regime = regime;
        
        state->head = (state->head + 1) % HAWKES_MAX_EVENTS;
        if (state->count < HAWKES_MAX_EVENTS) state->count++;
        state->total_events++;
        
        /* Immediate intensity jump from self-excitation */
        float mark_impact = compute_mark_impact(cfg, regime, abs_ret);
        state->intensity += rp->alpha * mark_impact;
        state->intensity = clampf(state->intensity, cfg->min_intensity, cfg->max_intensity);
        
        state->cache_valid = 0;
    }
    
    /* Update event rate EMA */
    float is_event = (abs_ret > rp->threshold) ? 1.0f : 0.0f;
    state->event_rate_ema = ema_alpha * is_event + 
                           (1.0f - ema_alpha) * state->event_rate_ema;
    
    return state->intensity;
}

float hawkes_get_intensity(const HawkesState *state)
{
    return state ? state->intensity : 0.0f;
}

float hawkes_predict_intensity(const HawkesState *state, float future_time)
{
    if (!state) return 0.0f;
    
    const HawkesConfig *cfg = &state->config;
    int regime = state->current_regime;
    const HawkesRegimeParams *rp = &cfg->regime[regime];
    
    /* Baseline at future time */
    float pred = rp->mu;
    
    /* Decay of current excitation */
    for (int i = 0; i < state->count; i++) {
        int idx = (state->head - 1 - i + HAWKES_MAX_EVENTS) % HAWKES_MAX_EVENTS;
        const HawkesEvent *ev = &state->events[idx];
        
        float event_dt = future_time - ev->time;
        if (event_dt <= 0) continue;
        
        int ev_regime = ev->regime;
        const HawkesRegimeParams *ev_rp = &cfg->regime[ev_regime];
        
        float kernel = compute_kernel(cfg, ev_regime, event_dt);
        float mark_impact = compute_mark_impact(cfg, ev_regime, ev->mark);
        
        pred += ev_rp->alpha * kernel * mark_impact;
    }
    
    return clampf(pred, cfg->min_intensity, cfg->max_intensity);
}

/*═══════════════════════════════════════════════════════════════════════════
 * RBPF INTEGRATION
 *═══════════════════════════════════════════════════════════════════════════*/

void hawkes_modify_transition_probs(const HawkesState *state,
                                    float *trans_row,
                                    int current_regime,
                                    int n_regimes)
{
    if (!state || !trans_row) return;
    
    const HawkesConfig *cfg = &state->config;
    if (!cfg->modify_transitions) return;
    
    float intensity = state->intensity;
    float threshold = cfg->intensity_threshold;
    float max_boost = cfg->max_transition_boost;
    
    /* Only modify if intensity is significantly above/below threshold */
    float deviation = intensity - threshold;
    if (fabsf(deviation) < 0.05f) return;
    
    /* Compute boost magnitude (sigmoid-like saturation) */
    float boost = max_boost * tanhf(deviation * 2.0f);
    
    if (boost > 0 && current_regime < n_regimes - 1) {
        /* High intensity: boost upward transition */
        float steal = minf(boost, trans_row[current_regime] * 0.5f);
        trans_row[current_regime] -= steal;
        trans_row[current_regime + 1] += steal;
    } else if (boost < 0 && current_regime > 0) {
        /* Low intensity: boost downward transition */
        float steal = minf(-boost, trans_row[current_regime] * 0.5f);
        trans_row[current_regime] -= steal;
        trans_row[current_regime - 1] += steal;
    }
    
    /* Ensure valid probabilities */
    float sum = 0.0f;
    for (int j = 0; j < n_regimes; j++) {
        trans_row[j] = maxf(0.0f, trans_row[j]);
        sum += trans_row[j];
    }
    if (sum > 0) {
        for (int j = 0; j < n_regimes; j++) {
            trans_row[j] /= sum;
        }
    }
}

float hawkes_regime_loglik(const HawkesState *state, int regime, float vol)
{
    if (!state) return 0.0f;
    
    const HawkesConfig *cfg = &state->config;
    regime = clampf(regime, 0, cfg->n_regimes - 1);
    
    const HawkesRegimeParams *rp = &cfg->regime[regime];
    
    /* Expected intensity for this regime */
    float expected_intensity = rp->mu * (1.0f + hawkes_branching_ratio(state, regime));
    
    /* Compare current intensity to expected */
    float log_lik = 0.0f;
    
    /* Higher intensity should favor higher regimes */
    float intensity_ratio = state->intensity / (expected_intensity + 1e-6f);
    
    /* Penalize mismatch */
    float mismatch = logf(intensity_ratio);
    log_lik -= 0.5f * mismatch * mismatch;
    
    return log_lik;
}

float hawkes_expected_vol(const HawkesState *state, int regime)
{
    if (!state) return 0.01f;
    
    const HawkesConfig *cfg = &state->config;
    regime = clampf(regime, 0, cfg->n_regimes - 1);
    
    const HawkesRegimeParams *rp = &cfg->regime[regime];
    
    /* Map intensity to volatility */
    /* Rough mapping: intensity 0.1 ≈ 5% vol, intensity 0.5 ≈ 25% vol */
    float base_vol = rp->threshold * 2.5f;  /* Threshold as proxy for regime vol */
    float intensity_factor = state->intensity / (rp->mu + 1e-6f);
    
    return base_vol * sqrtf(intensity_factor);
}

/*═══════════════════════════════════════════════════════════════════════════
 * CALIBRATION
 *═══════════════════════════════════════════════════════════════════════════*/

void hawkes_calibrate_moments(const float *returns, int n, float threshold,
                              float *out_mu, float *out_alpha, float *out_beta)
{
    if (!returns || n < 100) {
        if (out_mu) *out_mu = 0.05f;
        if (out_alpha) *out_alpha = 0.5f;
        if (out_beta) *out_beta = 0.1f;
        return;
    }
    
    /* Count events */
    int n_events = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(returns[i]) > threshold) n_events++;
    }
    
    /* Empirical event rate */
    float lambda_hat = (float)n_events / n;
    
    /* Estimate inter-arrival times for clustering */
    float *arrivals = (float*)malloc(n_events * sizeof(float));
    int ev_idx = 0;
    for (int i = 0; i < n && ev_idx < n_events; i++) {
        if (fabsf(returns[i]) > threshold) {
            arrivals[ev_idx++] = (float)i;
        }
    }
    
    /* Mean and variance of inter-arrival times */
    float sum_iat = 0.0f, sum_iat_sq = 0.0f;
    for (int i = 1; i < n_events; i++) {
        float iat = arrivals[i] - arrivals[i-1];
        sum_iat += iat;
        sum_iat_sq += iat * iat;
    }
    free(arrivals);
    
    float mean_iat = (n_events > 1) ? sum_iat / (n_events - 1) : 1.0f;
    float var_iat = (n_events > 2) ? 
        (sum_iat_sq / (n_events - 1) - mean_iat * mean_iat) : mean_iat * mean_iat;
    
    /* Method of moments estimates */
    /* For Hawkes: E[IAT] = 1/(μ(1-n)), Var[IAT] = (1+n)/((1-n)² μ²) */
    /* where n = α/β is branching ratio */
    
    /* Coefficient of variation squared */
    float cv_sq = var_iat / (mean_iat * mean_iat + 1e-10f);
    
    /* Estimate branching ratio from CV */
    /* CV² = (1+n)/(1-n) for Hawkes → n = (CV² - 1)/(CV² + 1) */
    float n_hat = (cv_sq - 1.0f) / (cv_sq + 1.0f);
    n_hat = clampf(n_hat, 0.1f, 0.9f);
    
    /* Baseline intensity */
    float mu_hat = lambda_hat * (1.0f - n_hat);
    mu_hat = clampf(mu_hat, 0.01f, 0.5f);
    
    /* Beta from typical clustering timescale */
    /* Assume clustering decays over ~10 events worth of time */
    float beta_hat = 0.693f / (mean_iat * 2.0f);  /* Half-life = 2 * mean_iat */
    beta_hat = clampf(beta_hat, 0.02f, 0.5f);
    
    /* Alpha from branching ratio */
    float alpha_hat = n_hat * beta_hat;
    
    if (out_mu) *out_mu = mu_hat;
    if (out_alpha) *out_alpha = alpha_hat;
    if (out_beta) *out_beta = beta_hat;
}

int hawkes_calibrate_mle(const float *returns, int n, float threshold,
                         float *out_mu, float *out_alpha, float *out_beta)
{
    /* Initialize with method of moments */
    float mu, alpha, beta;
    hawkes_calibrate_moments(returns, n, threshold, &mu, &alpha, &beta);
    
    /* Simple gradient descent MLE */
    float lr = 0.001f;
    int max_iter = 100;
    
    /* Build event list */
    int n_events = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(returns[i]) > threshold) n_events++;
    }
    
    float *event_times = (float*)malloc(n_events * sizeof(float));
    float *event_marks = (float*)malloc(n_events * sizeof(float));
    int ev_idx = 0;
    for (int i = 0; i < n && ev_idx < n_events; i++) {
        if (fabsf(returns[i]) > threshold) {
            event_times[ev_idx] = (float)i;
            event_marks[ev_idx] = fabsf(returns[i]);
            ev_idx++;
        }
    }
    
    float T = (float)n;  /* Observation window */
    
    for (int iter = 0; iter < max_iter; iter++) {
        /* Compute log-likelihood and gradients */
        float ll = 0.0f;
        float d_mu = 0.0f, d_alpha = 0.0f, d_beta = 0.0f;
        
        /* Integral of intensity (compensator) */
        float compensator = mu * T;
        d_mu += T;
        
        for (int i = 0; i < n_events; i++) {
            float ti = event_times[i];
            
            /* Intensity at event time */
            float lambda_i = mu;
            for (int j = 0; j < i; j++) {
                float dt = ti - event_times[j];
                lambda_i += alpha * expf(-beta * dt);
            }
            
            /* Log-likelihood contribution */
            ll += logf(lambda_i + 1e-10f);
            
            /* Compensator contribution from this event */
            float remaining = T - ti;
            float decay_integral = (1.0f - expf(-beta * remaining)) / beta;
            compensator += alpha * decay_integral;
            
            /* Gradients */
            float inv_lambda = 1.0f / (lambda_i + 1e-10f);
            d_mu += inv_lambda;
            
            for (int j = 0; j < i; j++) {
                float dt = ti - event_times[j];
                float kernel = expf(-beta * dt);
                d_alpha += inv_lambda * kernel;
                d_beta -= inv_lambda * alpha * dt * kernel;
            }
            
            /* Compensator gradients */
            d_alpha -= decay_integral;
            d_beta -= alpha * (remaining * expf(-beta * remaining) - decay_integral) / beta;
        }
        
        ll -= compensator;
        
        /* Gradient ascent */
        mu += lr * (d_mu - n_events / (mu + 1e-10f));  /* Regularize */
        alpha += lr * d_alpha;
        beta += lr * d_beta;
        
        /* Project to valid region */
        mu = clampf(mu, 0.001f, 0.5f);
        alpha = clampf(alpha, 0.01f, beta * 0.95f);  /* Ensure subcritical */
        beta = clampf(beta, 0.01f, 1.0f);
    }
    
    free(event_times);
    free(event_marks);
    
    if (out_mu) *out_mu = mu;
    if (out_alpha) *out_alpha = alpha;
    if (out_beta) *out_beta = beta;
    
    return 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

float hawkes_branching_ratio(const HawkesState *state, int regime)
{
    if (!state) return 0.0f;
    
    regime = clampf(regime, 0, state->config.n_regimes - 1);
    const HawkesRegimeParams *rp = &state->config.regime[regime];
    
    return rp->alpha / (rp->beta + 1e-10f);
}

float hawkes_half_life(const HawkesState *state, int regime)
{
    if (!state) return 0.0f;
    
    regime = clampf(regime, 0, state->config.n_regimes - 1);
    const HawkesRegimeParams *rp = &state->config.regime[regime];
    
    return 0.693147f / (rp->beta + 1e-10f);
}

void hawkes_print_state(const HawkesState *state)
{
    if (!state) return;
    
    printf("\n");
    printf("+===========================================================+\n");
    printf("|              HAWKES STATE                                 |\n");
    printf("+===========================================================+\n");
    printf("| Time: %.1f  Regime: %d                                    \n", 
           state->current_time, state->current_regime);
    printf("| Intensity: %.4f (baseline=%.4f + excited=%.4f)           \n",
           state->intensity, state->intensity_baseline, state->intensity_excited);
    printf("| Intensity EMA: %.4f                                       \n", state->intensity_ema);
    printf("| Event rate EMA: %.4f                                      \n", state->event_rate_ema);
    printf("+-----------------------------------------------------------+\n");
    printf("| Events in buffer: %d / %d                                 \n", 
           state->count, HAWKES_MAX_EVENTS);
    printf("| Total events: %d                                          \n", state->total_events);
    printf("+===========================================================+\n");
}

void hawkes_print_config(const HawkesConfig *cfg)
{
    if (!cfg) return;
    
    printf("\n");
    printf("+===========================================================+\n");
    printf("|              HAWKES CONFIG                                |\n");
    printf("+===========================================================+\n");
    printf("| Regimes: %d  Kernel: %d  Mark: %d                         \n",
           cfg->n_regimes, cfg->kernel, cfg->mark_type);
    printf("+-----------------------------------------------------------+\n");
    printf("| Per-Regime Parameters:                                    |\n");
    printf("|   R    mu     alpha   beta    n=α/β   t½      thresh      |\n");
    for (int r = 0; r < cfg->n_regimes; r++) {
        const HawkesRegimeParams *rp = &cfg->regime[r];
        float n = rp->alpha / (rp->beta + 1e-10f);
        float half_life = 0.693147f / (rp->beta + 1e-10f);
        printf("|   %d   %.3f   %.3f   %.3f   %.2f   %.1f    %.3f        |\n",
               r, rp->mu, rp->alpha, rp->beta, n, half_life, rp->threshold);
    }
    printf("+-----------------------------------------------------------+\n");
    printf("| Transition modification: %s                              \n",
           cfg->modify_transitions ? "ON" : "OFF");
    printf("| Intensity threshold: %.2f  Max boost: %.2f                \n",
           cfg->intensity_threshold, cfg->max_transition_boost);
    printf("+===========================================================+\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * BATCH PROCESSING
 *═══════════════════════════════════════════════════════════════════════════*/

void hawkes_update_batch(HawkesState *state,
                         const float *returns,
                         const int *regimes,
                         int n,
                         float *out_intensity)
{
    if (!state || !returns) return;
    
    for (int t = 0; t < n; t++) {
        int regime = regimes ? regimes[t] : state->current_regime;
        float intensity = hawkes_update(state, state->current_time + 1.0f, returns[t], regime);
        
        if (out_intensity) {
            out_intensity[t] = intensity;
        }
    }
}
