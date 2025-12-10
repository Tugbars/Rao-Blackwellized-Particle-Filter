/**
 * @file rbpf_preprocess.h
 * @brief Production preprocessing layer for RBPF
 *
 * MANDATORY for intraday data:
 *   1. Zero/Small Return Handling - log(0) = -inf breaks everything
 *   2. Diurnal Adjustment - Removes predictable U-shape intraday pattern
 *   3. Outlier Pre-Clamping - Optional safety rail before Robust OCSN
 *
 * Without preprocessing, RBPF will:
 *   - Flag every market open as "crisis"
 *   - Flag every lunch lull as "regime change"
 *   - Explode on zero returns (common in illiquid instruments)
 *
 * Usage:
 *   RBPF_Preprocessor *prep = rbpf_preprocess_create(RBPF_PREPROCESS_EQUITY_US);
 *   rbpf_preprocess_init_diurnal(prep, historical_returns, n_days, ticks_per_day);
 *   
 *   // In hot path:
 *   rbpf_real_t adj_return = rbpf_preprocess(prep, raw_return, time_of_day);
 *   rbpf_ext_step(ext, adj_return, &output);
 */

#ifndef RBPF_PREPROCESS_H
#define RBPF_PREPROCESS_H

#include "rbpf_ksc.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * CONFIGURATION
 *============================================================================*/

#define RBPF_PREPROCESS_MAX_BINS    128     /* Max intraday time bins */
#define RBPF_PREPROCESS_DEFAULT_BINS 78     /* 5-min bins for 6.5hr US session */

/* Preset configurations */
typedef enum {
    RBPF_PREPROCESS_NONE = 0,       /* No preprocessing (already adjusted) */
    RBPF_PREPROCESS_EQUITY_US,      /* US equities: 9:30-16:00 ET */
    RBPF_PREPROCESS_EQUITY_EU,      /* EU equities: 8:00-16:30 */
    RBPF_PREPROCESS_FX_24H,         /* FX: 24h with Asian/London/NY sessions */
    RBPF_PREPROCESS_CRYPTO_24H,     /* Crypto: 24h, no clear pattern */
    RBPF_PREPROCESS_FUTURES_US,     /* US futures: extended hours */
    RBPF_PREPROCESS_CUSTOM          /* User-defined */
} RBPF_PreprocessPreset;

/*============================================================================
 * DIURNAL PATTERN
 *============================================================================*/

/**
 * Diurnal volatility pattern (intraday seasonality)
 *
 * The typical U-shape for US equities:
 *   - High vol at open (9:30-10:00)
 *   - Declining through morning
 *   - Low vol at lunch (12:00-13:00)
 *   - Rising through afternoon
 *   - High vol at close (15:30-16:00)
 *
 * We store sqrt(E[r²]) for each time bin, then divide raw returns by this
 * to get "time-adjusted" returns with roughly constant unconditional vol.
 */
typedef struct {
    int n_bins;                             /* Number of time bins */
    rbpf_real_t vol_factor[RBPF_PREPROCESS_MAX_BINS];  /* Relative vol per bin */
    rbpf_real_t inv_vol_factor[RBPF_PREPROCESS_MAX_BINS]; /* 1/vol for fast division */
    
    /* Session timing (minutes from midnight) */
    int session_start_min;      /* e.g., 570 for 9:30 */
    int session_end_min;        /* e.g., 960 for 16:00 */
    int bin_duration_min;       /* e.g., 5 for 5-minute bins */
    
    /* Statistics for online update */
    double sum_sq[RBPF_PREPROCESS_MAX_BINS];    /* Running sum of r² */
    int count[RBPF_PREPROCESS_MAX_BINS];        /* Count per bin */
    
    /* Smoothing */
    rbpf_real_t ema_alpha;      /* EMA smoothing for online updates */
    int warmup_days;            /* Days before pattern is trusted */
    int days_seen;              /* Days of data processed */
} RBPF_DiurnalPattern;

/*============================================================================
 * PREPROCESSOR STATE
 *============================================================================*/

typedef struct {
    /* Configuration */
    RBPF_PreprocessPreset preset;
    int enable_diurnal;         /* Apply diurnal adjustment */
    int enable_zero_clamp;      /* Clamp small returns */
    int enable_outlier_clamp;   /* Pre-clamp extreme returns */
    int enable_online_update;   /* Update diurnal pattern online */
    
    /* Zero handling */
    rbpf_real_t zero_floor;     /* Minimum |return| (default: 1e-8) */
    
    /* Outlier clamping (before Robust OCSN) */
    rbpf_real_t outlier_mult;   /* Clamp at this × running_vol (default: 15) */
    rbpf_real_t running_vol;    /* EMA of |returns| for clamping reference */
    rbpf_real_t vol_ema_alpha;  /* EMA alpha for running_vol */
    
    /* Diurnal pattern */
    RBPF_DiurnalPattern diurnal;
    
    /* Statistics */
    uint64_t n_zeros_clamped;
    uint64_t n_outliers_clamped;
    uint64_t n_total;
} RBPF_Preprocessor;

/*============================================================================
 * API
 *============================================================================*/

/**
 * Create preprocessor with preset configuration
 */
RBPF_Preprocessor *rbpf_preprocess_create(RBPF_PreprocessPreset preset);

/**
 * Destroy preprocessor
 */
void rbpf_preprocess_destroy(RBPF_Preprocessor *prep);

/**
 * Initialize diurnal pattern from historical data
 *
 * @param prep          Preprocessor
 * @param returns       Array of historical returns
 * @param timestamps    Array of timestamps (minutes from midnight)
 * @param n             Number of observations
 *
 * This computes the average vol for each time bin and stores
 * the relative scaling factors.
 */
void rbpf_preprocess_init_diurnal(
    RBPF_Preprocessor *prep,
    const rbpf_real_t *returns,
    const int *timestamps,      /* Minutes from midnight */
    int n);

/**
 * Initialize diurnal pattern with explicit factors
 *
 * @param prep          Preprocessor
 * @param vol_factors   Array of relative vol factors per bin
 * @param n_bins        Number of bins
 */
void rbpf_preprocess_set_diurnal(
    RBPF_Preprocessor *prep,
    const rbpf_real_t *vol_factors,
    int n_bins);

/**
 * Set default U-shape pattern (no historical data needed)
 *
 * Uses a parametric model:
 *   vol(t) = A * exp(-B * (t - t_open)²) + C + D * exp(-E * (t_close - t)²)
 *
 * Reasonable for US equities when you don't have historical data.
 */
void rbpf_preprocess_set_default_ushape(RBPF_Preprocessor *prep);

/**
 * MAIN FUNCTION: Preprocess a single return
 *
 * @param prep          Preprocessor
 * @param raw_return    Raw return value
 * @param time_min      Time of day in minutes from midnight (e.g., 570 = 9:30)
 * @return              Adjusted return suitable for RBPF
 *
 * This is the hot path function - optimized for low latency.
 */
static inline rbpf_real_t rbpf_preprocess(
    RBPF_Preprocessor *prep,
    rbpf_real_t raw_return,
    int time_min);

/**
 * Update running statistics (call after each observation)
 *
 * If enable_online_update is set, this updates the diurnal pattern
 * incrementally. Call this AFTER rbpf_preprocess().
 */
void rbpf_preprocess_update(
    RBPF_Preprocessor *prep,
    rbpf_real_t raw_return,
    int time_min);

/**
 * Signal new trading day (resets daily state if any)
 */
void rbpf_preprocess_new_day(RBPF_Preprocessor *prep);

/**
 * Get current diurnal factor for a time
 */
rbpf_real_t rbpf_preprocess_get_diurnal_factor(
    const RBPF_Preprocessor *prep,
    int time_min);

/**
 * Print configuration and statistics
 */
void rbpf_preprocess_print_stats(const RBPF_Preprocessor *prep);

/**
 * Print diurnal pattern (for visualization/debugging)
 */
void rbpf_preprocess_print_diurnal(const RBPF_Preprocessor *prep);

/*============================================================================
 * INLINE IMPLEMENTATION (Hot Path)
 *============================================================================*/

static inline int rbpf_preprocess_time_to_bin(
    const RBPF_Preprocessor *prep,
    int time_min)
{
    const RBPF_DiurnalPattern *d = &prep->diurnal;
    
    /* Clamp to session */
    if (time_min < d->session_start_min) time_min = d->session_start_min;
    if (time_min >= d->session_end_min) time_min = d->session_end_min - 1;
    
    int bin = (time_min - d->session_start_min) / d->bin_duration_min;
    if (bin < 0) bin = 0;
    if (bin >= d->n_bins) bin = d->n_bins - 1;
    
    return bin;
}

static inline rbpf_real_t rbpf_preprocess(
    RBPF_Preprocessor *prep,
    rbpf_real_t raw_return,
    int time_min)
{
    rbpf_real_t adj = raw_return;
    
    prep->n_total++;
    
    /* 1. Zero/small return handling - CRITICAL for log transform */
    if (prep->enable_zero_clamp) {
        rbpf_real_t abs_ret = rbpf_fabs(adj);
        if (abs_ret < prep->zero_floor) {
            /* Preserve sign, clamp magnitude */
            rbpf_real_t sign = (adj >= 0) ? RBPF_REAL(1.0) : RBPF_REAL(-1.0);
            adj = sign * prep->zero_floor;
            prep->n_zeros_clamped++;
        }
    }
    
    /* 2. Outlier pre-clamping (optional safety rail) */
    if (prep->enable_outlier_clamp && prep->running_vol > RBPF_REAL(1e-10)) {
        rbpf_real_t threshold = prep->outlier_mult * prep->running_vol;
        if (adj > threshold) {
            adj = threshold;
            prep->n_outliers_clamped++;
        } else if (adj < -threshold) {
            adj = -threshold;
            prep->n_outliers_clamped++;
        }
    }
    
    /* 3. Diurnal adjustment - remove U-shape */
    if (prep->enable_diurnal && prep->diurnal.days_seen >= prep->diurnal.warmup_days) {
        int bin = rbpf_preprocess_time_to_bin(prep, time_min);
        adj *= prep->diurnal.inv_vol_factor[bin];
    }
    
    /* Update running vol for next iteration */
    if (prep->enable_outlier_clamp) {
        rbpf_real_t abs_raw = rbpf_fabs(raw_return);
        prep->running_vol = prep->vol_ema_alpha * abs_raw +
                           (RBPF_REAL(1.0) - prep->vol_ema_alpha) * prep->running_vol;
    }
    
    return adj;
}

/*============================================================================
 * UTILITY: Time conversion helpers
 *============================================================================*/

/**
 * Convert HH:MM to minutes from midnight
 */
static inline int rbpf_time_to_minutes(int hour, int minute)
{
    return hour * 60 + minute;
}

/**
 * Convert Unix timestamp to minutes from midnight (UTC)
 * For local time, adjust timezone before calling.
 */
static inline int rbpf_unix_to_minutes(int64_t unix_sec)
{
    return (int)((unix_sec % 86400) / 60);
}

#ifdef __cplusplus
}
#endif

#endif /* RBPF_PREPROCESS_H */
