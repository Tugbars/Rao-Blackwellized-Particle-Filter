/**
 * @file rbpf_ocsn_robust.c
 * @brief Robust OCSN Implementation - 11th Outlier Component
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THEORY
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Standard OCSN (Omori et al., 2007) uses 10-component mixture to approximate
 * log(χ²₁) for the observation equation:
 *
 *   r_t = exp(h_t/2) * ε_t,  ε_t ~ N(0,1)
 *   y_t = log(r_t²) = h_t + log(ε_t²)
 *
 * In the KSC parameterization used by rbpf_ksc.c, we store ℓ = h/2, so:
 *   y_t = 2*ℓ_t + log(ε_t²)
 *
 * This gives observation equation: y = H*ℓ + ξ, where H=2 and ξ ~ log(χ²₁)
 *
 * The 10-component mixture works well for normal returns (|ε| < 4σ), but
 * fat-tail events (8-15σ) cause particle collapse because:
 *   - All particles assign near-zero likelihood
 *   - ESS crashes to single digits
 *   - Resampling produces degenerate ensemble
 *
 * ROBUST OCSN adds an 11th "outlier" component:
 *
 *   P(obs | h, regime) = (1 - π_out) × P_OCSN(obs | h)
 *                      + π_out × N(obs | H*h, H²*P + σ²_out)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * CRITICAL: H = 2 OBSERVATION MATRIX
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The observation equation y = H*h + ξ has H=2, NOT H=1!
 *
 * This affects ALL Kalman calculations:
 *   - Innovation mean:     y - H*h - m_k = y - 2*h - m_k
 *   - Innovation variance: H²*P + v_k = 4*P + v_k
 *   - Kalman gain:         K = H*P / S = 2*P / S
 *   - State update:        h_post = h_prior + K * innovation
 *   - Variance update:     P_post = (1 - K*H) * P = (1 - 2*K) * P
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "rbpf_ksc.h"
#include <math.h>
#include <float.h>

/*═══════════════════════════════════════════════════════════════════════════
 * OBSERVATION EQUATION CONSTANTS
 *═══════════════════════════════════════════════════════════════════════════*/

#define H_OBS RBPF_REAL(2.0)  /* Observation matrix: y = H*h + noise */
#define H2_OBS RBPF_REAL(4.0) /* H² for variance calculations */

/*═══════════════════════════════════════════════════════════════════════════
 * OCSN MIXTURE PARAMETERS (Kim, Shephard, Chib 1998 / Omori 2007)
 *
 * 10-component mixture approximation to log(χ²₁)
 *═══════════════════════════════════════════════════════════════════════════*/

static const rbpf_real_t OCSN_PROB[10] = {
    0.00609f, 0.04775f, 0.13057f, 0.20674f, 0.22715f,
    0.18842f, 0.12047f, 0.05591f, 0.01575f, 0.00115f};

static const rbpf_real_t OCSN_MEAN[10] = {
    1.92677f, 1.34744f, 0.73504f, 0.02266f, -0.85173f,
    -1.97278f, -3.46788f, -5.55246f, -8.68384f, -14.65000f};

static const rbpf_real_t OCSN_VAR[10] = {
    0.11265f, 0.17788f, 0.26768f, 0.40611f, 0.62699f,
    0.98583f, 1.57469f, 2.54498f, 4.16591f, 7.33342f};

/* Precomputed: log(π_k) for each component */
static const rbpf_real_t OCSN_LOG_PROB[10] = {
    -5.101f, -3.042f, -2.036f, -1.577f, -1.482f,
    -1.669f, -2.116f, -2.884f, -4.151f, -6.768f};

/* Precomputed: -0.5 * log(2π) */
#define LOG_2PI_HALF RBPF_REAL(-0.9189385332)

/*═══════════════════════════════════════════════════════════════════════════
 * HELPER: Log-Sum-Exp for numerical stability
 *═══════════════════════════════════════════════════════════════════════════*/

static inline rbpf_real_t log_sum_exp_2(rbpf_real_t log_a, rbpf_real_t log_b)
{
    if (log_a > log_b)
    {
        return log_a + rbpf_log(1.0f + rbpf_exp(log_b - log_a));
    }
    else
    {
        return log_b + rbpf_log(1.0f + rbpf_exp(log_a - log_b));
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * rbpf_ksc_update_robust
 *
 * Performs RBPF update with 11-component Robust OCSN likelihood.
 * Matches rbpf_ksc_update() exactly, but adds outlier component.
 *
 * @param rbpf          RBPF filter state
 * @param y             Transformed observation: y = log(r²)
 * @param robust_ocsn   Robust OCSN configuration (per-regime outlier params)
 * @return              Weighted average log-likelihood
 *═══════════════════════════════════════════════════════════════════════════*/

rbpf_real_t rbpf_ksc_update_robust(
    RBPF_KSC *rbpf,
    rbpf_real_t y,
    const RBPF_RobustOCSN *robust_ocsn)
{
    if (!rbpf || !robust_ocsn || !robust_ocsn->enabled)
    {
        return rbpf_ksc_update(rbpf, y);
    }

    const int n = rbpf->n_particles;
    const rbpf_real_t NEG_HALF = RBPF_REAL(-0.5);

    rbpf_real_t *restrict mu_pred = rbpf->mu_pred;
    rbpf_real_t *restrict var_pred = rbpf->var_pred;
    rbpf_real_t *restrict mu = rbpf->mu;
    rbpf_real_t *restrict var = rbpf->var;
    rbpf_real_t *restrict log_weight = rbpf->log_weight;
    const int *restrict regime = rbpf->regime;

    /* Temporary storage for log-likelihoods (for log-sum-exp) */
    rbpf_real_t log_liks[11]; /* 10 OCSN + 1 outlier */

    /* Accumulator for marginal likelihood */
    rbpf_real_t total_marginal = RBPF_REAL(0.0);

    for (int i = 0; i < n; i++)
    {
        int r = regime[i];
        rbpf_real_t h_prior = mu_pred[i];
        rbpf_real_t P_prior = var_pred[i];

        /* Get outlier parameters for this regime */
        rbpf_real_t pi_out = robust_ocsn->regime[r].prob;
        rbpf_real_t var_out = robust_ocsn->regime[r].variance;

        /* Clamp outlier variance to safe bounds */
        if (var_out < RBPF_OUTLIER_VAR_MIN)
            var_out = RBPF_OUTLIER_VAR_MIN;
        if (var_out > RBPF_OUTLIER_VAR_MAX)
            var_out = RBPF_OUTLIER_VAR_MAX;

        rbpf_real_t log_1_minus_pi = rbpf_log(1.0f - pi_out);
        rbpf_real_t log_pi = rbpf_log(pi_out);

        /*───────────────────────────────────────────────────────────────────
         * Pass 1: Compute log-likelihoods for all 11 components
         *
         * OCSN components k=0..9:
         *   y ~ N(H*h + m_k, H²*P + v_k)  with H=2
         *
         * Outlier component k=10:
         *   y ~ N(H*h, H²*P + var_out)
         *─────────────────────────────────────────────────────────────────*/

        rbpf_real_t max_log_lik = RBPF_REAL(-1e30);

        /* 10 OCSN components */
        for (int k = 0; k < 10; k++)
        {
            rbpf_real_t y_adj = y - OCSN_MEAN[k];
            rbpf_real_t innov = y_adj - H_OBS * h_prior;
            rbpf_real_t S = H2_OBS * P_prior + OCSN_VAR[k];
            rbpf_real_t innov2_S = innov * innov / S;

            /* log N(y | H*h + m_k, S) + log(π_k) + log(1 - π_out) */
            log_liks[k] = log_1_minus_pi + OCSN_LOG_PROB[k] +
                          NEG_HALF * (rbpf_log(S) + innov2_S);

            if (log_liks[k] > max_log_lik)
                max_log_lik = log_liks[k];
        }

        /* Outlier component */
        {
            rbpf_real_t innov = y - H_OBS * h_prior;
            rbpf_real_t S = H2_OBS * P_prior + var_out;
            rbpf_real_t innov2_S = innov * innov / S;

            /* log N(y | H*h, S) + log(π_out) */
            log_liks[10] = log_pi + NEG_HALF * (rbpf_log(S) + innov2_S);

            if (log_liks[10] > max_log_lik)
                max_log_lik = log_liks[10];
        }

        /*───────────────────────────────────────────────────────────────────
         * Pass 2: Normalize and compute weighted Kalman update
         *
         * Total likelihood = sum of all 11 component likelihoods
         * Each component contributes:
         *   - Likelihood weight w_k
         *   - Posterior mean h_k
         *   - Posterior variance P_k
         *
         * Final posterior uses law of total variance:
         *   E[h] = Σ w_k * h_k
         *   E[h²] = Σ w_k * (P_k + h_k²)
         *   Var[h] = E[h²] - E[h]²
         *─────────────────────────────────────────────────────────────────*/

        rbpf_real_t lik_total = RBPF_REAL(0.0);
        rbpf_real_t h_accum = RBPF_REAL(0.0);
        rbpf_real_t h2_accum = RBPF_REAL(0.0); /* E[h²] = Σ w_k (P_k + h_k²) */

        /* 10 OCSN components */
        for (int k = 0; k < 10; k++)
        {
            rbpf_real_t lik = rbpf_exp(log_liks[k] - max_log_lik);

            /* Kalman update for component k */
            rbpf_real_t y_adj = y - OCSN_MEAN[k];
            rbpf_real_t innov = y_adj - H_OBS * h_prior;
            rbpf_real_t S = H2_OBS * P_prior + OCSN_VAR[k];
            rbpf_real_t K = H_OBS * P_prior / S;

            rbpf_real_t h_k = h_prior + K * innov;
            rbpf_real_t P_k = (RBPF_REAL(1.0) - K * H_OBS) * P_prior;

            lik_total += lik;
            h_accum += lik * h_k;
            h2_accum += lik * (P_k + h_k * h_k);
        }

        /* Outlier component */
        {
            rbpf_real_t lik = rbpf_exp(log_liks[10] - max_log_lik);

            /* Kalman update for outlier component */
            rbpf_real_t innov = y - H_OBS * h_prior;
            rbpf_real_t S = H2_OBS * P_prior + var_out;
            rbpf_real_t K = H_OBS * P_prior / S;

            rbpf_real_t h_out = h_prior + K * innov;
            rbpf_real_t P_out = (RBPF_REAL(1.0) - K * H_OBS) * P_prior;

            lik_total += lik;
            h_accum += lik * h_out;
            h2_accum += lik * (P_out + h_out * h_out);
        }

        /*───────────────────────────────────────────────────────────────────
         * Finalize posterior
         *─────────────────────────────────────────────────────────────────*/

        rbpf_real_t inv_lik = RBPF_REAL(1.0) / (lik_total + RBPF_REAL(1e-30));
        rbpf_real_t h_post = h_accum * inv_lik;
        rbpf_real_t E_h2 = h2_accum * inv_lik;
        rbpf_real_t P_post = E_h2 - h_post * h_post;

        /* Floor variance */
        if (P_post < RBPF_REAL(1e-6))
            P_post = RBPF_REAL(1e-6);

        /* Store posterior */
        mu[i] = h_post;
        var[i] = P_post;

        /* Update log-weight: log(sum * exp(max)) = log(sum) + max */
        log_weight[i] += rbpf_log(lik_total + RBPF_REAL(1e-30)) + max_log_lik;

        /* Accumulate marginal likelihood: lik_total * exp(max_log_lik) */
        total_marginal += lik_total * rbpf_exp(max_log_lik);
    }

    /*
     * NOTE: We do NOT normalize w_norm here - that's done in rbpf_ksc_compute_outputs()
     * Return marginal likelihood (average observation likelihood across particles)
     */
    return total_marginal / n;
}

/*═══════════════════════════════════════════════════════════════════════════
 * rbpf_ksc_compute_outlier_fraction
 *
 * Computes the weighted average posterior probability that the observation
 * came from the outlier component (across all particles).
 *
 * @param rbpf          RBPF filter state
 * @param y             Transformed observation: y = log(r²)
 * @param robust_ocsn   Robust OCSN configuration
 * @return              Weighted average P(outlier | obs) in [0, 1]
 *═══════════════════════════════════════════════════════════════════════════*/

rbpf_real_t rbpf_ksc_compute_outlier_fraction(
    const RBPF_KSC *rbpf,
    rbpf_real_t y,
    const RBPF_RobustOCSN *robust_ocsn)
{
    if (!rbpf || !robust_ocsn || !robust_ocsn->enabled)
    {
        return RBPF_REAL(0.0);
    }

    const int n = rbpf->n_particles;
    const rbpf_real_t NEG_HALF = RBPF_REAL(-0.5);

    const rbpf_real_t *mu_pred = rbpf->mu_pred;
    const rbpf_real_t *var_pred = rbpf->var_pred;
    const rbpf_real_t *w_norm = rbpf->w_norm;
    const int *regime = rbpf->regime;

    rbpf_real_t weighted_sum = RBPF_REAL(0.0);

    for (int i = 0; i < n; i++)
    {
        int r = regime[i];
        rbpf_real_t h = mu_pred[i];
        rbpf_real_t P = var_pred[i];
        rbpf_real_t w = w_norm[i];

        /* Get outlier parameters */
        rbpf_real_t pi_out = robust_ocsn->regime[r].prob;
        rbpf_real_t var_out = robust_ocsn->regime[r].variance;

        if (var_out < RBPF_OUTLIER_VAR_MIN)
            var_out = RBPF_OUTLIER_VAR_MIN;
        if (var_out > RBPF_OUTLIER_VAR_MAX)
            var_out = RBPF_OUTLIER_VAR_MAX;

        rbpf_real_t log_1_minus_pi = rbpf_log(1.0f - pi_out);
        rbpf_real_t log_pi = rbpf_log(pi_out);

        /* Compute log-likelihoods for all components */
        rbpf_real_t max_log_lik = RBPF_REAL(-1e30);
        rbpf_real_t log_liks[11];

        /* OCSN components */
        for (int k = 0; k < 10; k++)
        {
            rbpf_real_t innov = y - OCSN_MEAN[k] - H_OBS * h;
            rbpf_real_t S = H2_OBS * P + OCSN_VAR[k];

            log_liks[k] = log_1_minus_pi + OCSN_LOG_PROB[k] +
                          NEG_HALF * (rbpf_log(S) + innov * innov / S);

            if (log_liks[k] > max_log_lik)
                max_log_lik = log_liks[k];
        }

        /* Outlier component */
        {
            rbpf_real_t innov = y - H_OBS * h;
            rbpf_real_t S = H2_OBS * P + var_out;

            log_liks[10] = log_pi + NEG_HALF * (rbpf_log(S) + innov * innov / S);

            if (log_liks[10] > max_log_lik)
                max_log_lik = log_liks[10];
        }

        /* Compute posterior probability of outlier */
        rbpf_real_t sum_lik = RBPF_REAL(0.0);
        rbpf_real_t lik_out = RBPF_REAL(0.0);

        for (int k = 0; k < 11; k++)
        {
            rbpf_real_t lik = rbpf_exp(log_liks[k] - max_log_lik);
            sum_lik += lik;
            if (k == 10)
                lik_out = lik;
        }

        rbpf_real_t post_out = lik_out / (sum_lik + RBPF_REAL(1e-30));

        /* Clamp */
        if (post_out < RBPF_REAL(0.0))
            post_out = RBPF_REAL(0.0);
        if (post_out > RBPF_REAL(1.0))
            post_out = RBPF_REAL(1.0);

        weighted_sum += w * post_out;
    }

    return weighted_sum;
}