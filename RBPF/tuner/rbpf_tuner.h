/**
 * @file rbpf_tuner.h
 * @brief RBPF Parameter Auto-Tuner
 *
 * Grid search over key RBPF parameters to maximize regime classification
 * accuracy. Searches over:
 *   - mu_vol separation (regime volatility levels)
 *   - Transition matrix stickiness
 *   - Hysteresis parameters (hold threshold, probability threshold)
 *
 * Usage:
 *   1. Prepare test data with known true regimes
 *   2. Initialize tuner with search configuration
 *   3. Run grid search (or manual evaluation)
 *   4. Apply best parameters to RBPF and Storvik
 *
 * Typical runtime: 5-15 minutes for full grid search (depends on data length)
 */

#ifndef RBPF_TUNER_H
#define RBPF_TUNER_H

#include "rbpf_ksc.h"
#include "rbpf_param_learn.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /*═══════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════*/

#define TUNER_MAX_REGIMES 8
#define TUNER_MAX_GRID_POINTS 10

    /**
     * Tuner search configuration
     */
    typedef struct
    {
        /*───────────────────────────────────────────────────────────────────────
         * REGIME SEPARATION SEARCH
         *
         * mu_vol values are computed as:
         *   mu_vol[0] = mu_vol_base
         *   mu_vol[r] = mu_vol[r-1] + gap
         *
         * Gap is searched between gap_min and gap_max.
         *───────────────────────────────────────────────────────────────────────*/
        float mu_vol_base;    /* Lowest regime mean (R0), e.g., -5.5 */
        float mu_vol_gap_min; /* Minimum gap between regimes, e.g., 1.0 */
        float mu_vol_gap_max; /* Maximum gap between regimes, e.g., 2.0 */
        int mu_vol_gap_steps; /* Grid points for gap search */

        /* Asymmetric gap control (prevents R3 from going unrealistically high) */
        int use_asymmetric_gaps; /* 1 = tighter gap for high regimes */
        float mu_vol_r3_max;     /* Maximum mu_vol for R3 (e.g., -0.3) */
        float gap_scale_high;    /* Scale factor for R2→R3 gap (e.g., 0.8) */

        /*───────────────────────────────────────────────────────────────────────
         * SIGMA_VOL SEARCH (optional - can be derived from mu_vol)
         *───────────────────────────────────────────────────────────────────────*/
        int search_sigma_vol;      /* 0 = derive from gap, 1 = independent search */
        float sigma_vol_base;      /* Base sigma for R0 */
        float sigma_vol_scale_min; /* Multiplier min per regime step */
        float sigma_vol_scale_max; /* Multiplier max per regime step */
        int sigma_vol_steps;

        /*───────────────────────────────────────────────────────────────────────
         * TRANSITION MATRIX SEARCH
         *───────────────────────────────────────────────────────────────────────*/
        float self_trans_min; /* Minimum self-transition prob, e.g., 0.95 */
        float self_trans_max; /* Maximum self-transition prob, e.g., 0.995 */
        int self_trans_steps; /* Grid points for self-transition */

        int adjacent_only; /* 1 = only allow R[i] -> R[i±1] transitions */

        /*───────────────────────────────────────────────────────────────────────
         * HYSTERESIS SEARCH
         *───────────────────────────────────────────────────────────────────────*/
        int hold_min;   /* Minimum hold threshold (ticks) */
        int hold_max;   /* Maximum hold threshold (ticks) */
        int hold_steps; /* Grid points for hold threshold */

        float prob_thresh_min; /* Minimum probability threshold */
        float prob_thresh_max; /* Maximum probability threshold */
        int prob_thresh_steps; /* Grid points for prob threshold */

        /*───────────────────────────────────────────────────────────────────────
         * THETA (mean reversion) SEARCH - usually fixed
         *───────────────────────────────────────────────────────────────────────*/
        int search_theta;      /* 0 = use defaults, 1 = search */
        float theta_base;      /* Base theta for R0 */
        float theta_increment; /* Increment per regime */

        /*───────────────────────────────────────────────────────────────────────
         * EVALUATION SETTINGS
         *───────────────────────────────────────────────────────────────────────*/
        int n_particles;        /* Particles per evaluation (e.g., 512) */
        int n_regimes;          /* Number of regimes (e.g., 4) */
        int n_eval_runs;        /* Runs to average (reduces variance) */
        int warmup_ticks;       /* Ticks before measuring accuracy */
        unsigned int base_seed; /* Base RNG seed for reproducibility */

        /*───────────────────────────────────────────────────────────────────────
         * OBJECTIVE FUNCTION WEIGHTS
         *───────────────────────────────────────────────────────────────────────*/
        float weight_accuracy;       /* Overall regime accuracy weight */
        float weight_middle_regimes; /* Bonus weight for R1, R2 accuracy */
        float weight_stability;      /* Penalty weight for regime switches */
        float weight_vol_rmse;       /* Penalty weight for volatility error */

        /*───────────────────────────────────────────────────────────────────────
         * STORVIK INTEGRATION
         *───────────────────────────────────────────────────────────────────────*/
        int use_storvik;              /* 1 = enable Storvik learning during eval */
        float storvik_prior_strength; /* Prior strength for Storvik */

        /*───────────────────────────────────────────────────────────────────────
         * OUTPUT CONTROL
         *───────────────────────────────────────────────────────────────────────*/
        int verbose;       /* 0 = quiet, 1 = progress, 2 = detailed */
        int print_every_n; /* Print progress every N evaluations */

    } TunerConfig;

    /**
     * Result of a single parameter evaluation
     */
    typedef struct
    {
        /*───────────────────────────────────────────────────────────────────────
         * PARAMETER VALUES
         *───────────────────────────────────────────────────────────────────────*/
        float mu_vol[TUNER_MAX_REGIMES];
        float sigma_vol[TUNER_MAX_REGIMES];
        float theta[TUNER_MAX_REGIMES];
        float self_trans;
        int hold_threshold;
        float prob_threshold;

        /*───────────────────────────────────────────────────────────────────────
         * ACCURACY METRICS
         *───────────────────────────────────────────────────────────────────────*/
        float overall_accuracy;                                /* Total correct / total */
        float regime_accuracy[TUNER_MAX_REGIMES];              /* Per-regime accuracy */
        float confusion[TUNER_MAX_REGIMES][TUNER_MAX_REGIMES]; /* Confusion matrix (row=true, col=pred) */

        /*───────────────────────────────────────────────────────────────────────
         * STABILITY METRICS
         *───────────────────────────────────────────────────────────────────────*/
        int n_switches;    /* Number of regime switches */
        float switch_rate; /* Switches per tick */

        /*───────────────────────────────────────────────────────────────────────
         * VOLATILITY METRICS
         *───────────────────────────────────────────────────────────────────────*/
        float vol_rmse; /* RMS error vs true volatility */
        float vol_bias; /* Mean bias (positive = overestimate) */

        /*───────────────────────────────────────────────────────────────────────
         * COMBINED OBJECTIVE
         *───────────────────────────────────────────────────────────────────────*/
        float objective; /* Weighted combination for optimization */

        /*───────────────────────────────────────────────────────────────────────
         * METADATA
         *───────────────────────────────────────────────────────────────────────*/
        int eval_id;      /* Evaluation index */
        float elapsed_ms; /* Time for this evaluation */

    } TunerResult;

    /**
     * Main tuner state
     */
    typedef struct
    {
        TunerConfig config;
        TunerResult best;
        TunerResult current;

        /*───────────────────────────────────────────────────────────────────────
         * TEST DATA
         *───────────────────────────────────────────────────────────────────────*/
        float *returns;    /* [n_ticks] observed returns */
        int *true_regimes; /* [n_ticks] ground truth regime labels */
        float *true_vol;   /* [n_ticks] ground truth volatility (optional) */
        int n_ticks;

        /*───────────────────────────────────────────────────────────────────────
         * PROGRESS TRACKING
         *───────────────────────────────────────────────────────────────────────*/
        int total_evals;
        int completed_evals;
        double total_elapsed_ms;

        /*───────────────────────────────────────────────────────────────────────
         * INTERNAL STATE
         *───────────────────────────────────────────────────────────────────────*/
        int initialized;

    } RBPFTuner;

    /*═══════════════════════════════════════════════════════════════════════════
     * DEFAULT CONFIGURATIONS
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Get default tuner configuration
     * Good starting point for 4-regime system
     */
    TunerConfig tuner_config_defaults(void);

    /**
     * Fast search configuration (fewer grid points)
     * Use for quick exploration before detailed search
     */
    TunerConfig tuner_config_fast(void);

    /**
     * Detailed search configuration (more grid points)
     * Use for final optimization
     */
    TunerConfig tuner_config_detailed(void);

    /**
     * Focus on middle regimes (R1, R2)
     * Use when R0/R3 are already accurate
     */
    TunerConfig tuner_config_focus_middle(void);

    /**
     * Balanced configuration with asymmetric gaps
     * Prevents extreme regimes from being sacrificed
     * Keeps R3 at realistic volatility levels
     */
    TunerConfig tuner_config_balanced(void);

    /*═══════════════════════════════════════════════════════════════════════════
     * LIFECYCLE
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Initialize tuner with test data
     *
     * @param tuner      Tuner state to initialize
     * @param cfg        Search configuration (NULL for defaults)
     * @param returns    Array of observed returns [n_ticks]
     * @param true_regimes Array of ground truth regimes [n_ticks]
     * @param true_vol   Array of true volatility (optional, can be NULL)
     * @param n_ticks    Number of ticks in test data
     * @return 0 on success, -1 on error
     */
    int tuner_init(RBPFTuner *tuner, const TunerConfig *cfg,
                   const float *returns, const int *true_regimes,
                   const float *true_vol, int n_ticks);

    /**
     * Free tuner resources
     */
    void tuner_free(RBPFTuner *tuner);

    /*═══════════════════════════════════════════════════════════════════════════
     * SEARCH METHODS
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Run full grid search
     *
     * Searches all combinations of parameters defined in config.
     * Updates tuner->best with best result found.
     *
     * @param tuner Initialized tuner
     * @return Best objective value found
     */
    float tuner_grid_search(RBPFTuner *tuner);

    /**
     * Evaluate single parameter set
     *
     * Useful for manual exploration or custom search algorithms.
     *
     * @param tuner   Initialized tuner
     * @param result  Parameters to evaluate (input) and results (output)
     * @return Objective value
     */
    float tuner_evaluate(RBPFTuner *tuner, TunerResult *result);

    /**
     * Evaluate with specific parameters (convenience function)
     */
    float tuner_evaluate_params(RBPFTuner *tuner,
                                const float *mu_vol,
                                const float *sigma_vol,
                                const float *theta,
                                float self_trans,
                                int hold_threshold,
                                float prob_threshold,
                                TunerResult *result);

    /*═══════════════════════════════════════════════════════════════════════════
     * RESULTS
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Get best result found so far
     */
    const TunerResult *tuner_get_best(const RBPFTuner *tuner);

    /**
     * Apply best parameters to RBPF instance
     */
    void tuner_apply_to_rbpf(const TunerResult *result, RBPF_KSC *rbpf, int n_regimes);

    /**
     * Apply best parameters to Storvik learner
     */
    void tuner_apply_to_storvik(const TunerResult *result, ParamLearner *learner, int n_regimes);

    /**
     * Generate C code snippet for best parameters
     * Writes to provided buffer (should be at least 2048 bytes)
     */
    void tuner_generate_code(const TunerResult *result, int n_regimes, char *buffer, int buffer_size);

    /*═══════════════════════════════════════════════════════════════════════════
     * OUTPUT
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Print result summary
     */
    void tuner_print_result(const TunerResult *result, int n_regimes);

    /**
     * Print confusion matrix
     */
    void tuner_print_confusion(const TunerResult *result, int n_regimes);

    /**
     * Print search progress
     */
    void tuner_print_progress(const RBPFTuner *tuner);

    /**
     * Export results to CSV
     */
    int tuner_export_csv(const RBPFTuner *tuner, const char *filename);

#ifdef __cplusplus
}
#endif

#endif /* RBPF_TUNER_H */