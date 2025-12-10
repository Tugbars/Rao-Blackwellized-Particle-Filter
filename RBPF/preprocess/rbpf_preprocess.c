/**
 * @file rbpf_preprocess.c
 * @brief Production preprocessing layer for RBPF
 */

#include "rbpf_preprocess.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/*============================================================================
 * PRESET CONFIGURATIONS
 *============================================================================*/

static void apply_preset_equity_us(RBPF_Preprocessor *prep)
{
    /* US Equities: 9:30 - 16:00 ET (390 minutes) */
    prep->diurnal.session_start_min = 570;   /* 9:30 */
    prep->diurnal.session_end_min = 960;     /* 16:00 */
    prep->diurnal.bin_duration_min = 5;      /* 5-minute bins */
    prep->diurnal.n_bins = 78;               /* 390 / 5 */
    prep->diurnal.warmup_days = 5;
    prep->diurnal.ema_alpha = 0.1f;
    
    prep->enable_diurnal = 1;
    prep->enable_zero_clamp = 1;
    prep->enable_outlier_clamp = 0;  /* Let Robust OCSN handle this */
    prep->enable_online_update = 1;
    
    prep->zero_floor = RBPF_REAL(1e-8);
    prep->outlier_mult = RBPF_REAL(15.0);
    prep->vol_ema_alpha = RBPF_REAL(0.01);
}

static void apply_preset_equity_eu(RBPF_Preprocessor *prep)
{
    /* EU Equities: 8:00 - 16:30 (510 minutes) */
    prep->diurnal.session_start_min = 480;   /* 8:00 */
    prep->diurnal.session_end_min = 990;     /* 16:30 */
    prep->diurnal.bin_duration_min = 5;
    prep->diurnal.n_bins = 102;
    prep->diurnal.warmup_days = 5;
    prep->diurnal.ema_alpha = 0.1f;
    
    prep->enable_diurnal = 1;
    prep->enable_zero_clamp = 1;
    prep->enable_outlier_clamp = 0;
    prep->enable_online_update = 1;
    
    prep->zero_floor = RBPF_REAL(1e-8);
    prep->outlier_mult = RBPF_REAL(15.0);
    prep->vol_ema_alpha = RBPF_REAL(0.01);
}

static void apply_preset_fx_24h(RBPF_Preprocessor *prep)
{
    /* FX: 24h with distinct sessions */
    prep->diurnal.session_start_min = 0;
    prep->diurnal.session_end_min = 1440;
    prep->diurnal.bin_duration_min = 15;     /* 15-min bins */
    prep->diurnal.n_bins = 96;
    prep->diurnal.warmup_days = 10;          /* More days for 24h */
    prep->diurnal.ema_alpha = 0.05f;
    
    prep->enable_diurnal = 1;
    prep->enable_zero_clamp = 1;
    prep->enable_outlier_clamp = 0;
    prep->enable_online_update = 1;
    
    prep->zero_floor = RBPF_REAL(1e-9);      /* FX has smaller moves */
    prep->outlier_mult = RBPF_REAL(12.0);    /* FX has fatter tails */
    prep->vol_ema_alpha = RBPF_REAL(0.005);
}

static void apply_preset_crypto_24h(RBPF_Preprocessor *prep)
{
    /* Crypto: 24h, less predictable pattern */
    prep->diurnal.session_start_min = 0;
    prep->diurnal.session_end_min = 1440;
    prep->diurnal.bin_duration_min = 30;     /* Coarser bins (less pattern) */
    prep->diurnal.n_bins = 48;
    prep->diurnal.warmup_days = 14;
    prep->diurnal.ema_alpha = 0.03f;
    
    prep->enable_diurnal = 1;                /* Still helps with Asian/US activity */
    prep->enable_zero_clamp = 1;
    prep->enable_outlier_clamp = 1;          /* Crypto needs pre-clamping */
    prep->enable_online_update = 1;
    
    prep->zero_floor = RBPF_REAL(1e-7);
    prep->outlier_mult = RBPF_REAL(10.0);    /* Crypto is wild */
    prep->vol_ema_alpha = RBPF_REAL(0.02);
}

static void apply_preset_futures_us(RBPF_Preprocessor *prep)
{
    /* US Futures: Extended hours (e.g., ES) 
     * Main session: 9:30-16:15, but trades nearly 24h */
    prep->diurnal.session_start_min = 0;
    prep->diurnal.session_end_min = 1440;
    prep->diurnal.bin_duration_min = 15;
    prep->diurnal.n_bins = 96;
    prep->diurnal.warmup_days = 7;
    prep->diurnal.ema_alpha = 0.08f;
    
    prep->enable_diurnal = 1;
    prep->enable_zero_clamp = 1;
    prep->enable_outlier_clamp = 0;
    prep->enable_online_update = 1;
    
    prep->zero_floor = RBPF_REAL(1e-8);
    prep->outlier_mult = RBPF_REAL(12.0);
    prep->vol_ema_alpha = RBPF_REAL(0.01);
}

/*============================================================================
 * CREATE / DESTROY
 *============================================================================*/

RBPF_Preprocessor *rbpf_preprocess_create(RBPF_PreprocessPreset preset)
{
    RBPF_Preprocessor *prep = (RBPF_Preprocessor *)calloc(1, sizeof(RBPF_Preprocessor));
    if (!prep) return NULL;
    
    prep->preset = preset;
    
    /* Initialize all vol factors to 1.0 (neutral) */
    for (int i = 0; i < RBPF_PREPROCESS_MAX_BINS; i++) {
        prep->diurnal.vol_factor[i] = RBPF_REAL(1.0);
        prep->diurnal.inv_vol_factor[i] = RBPF_REAL(1.0);
    }
    
    /* Apply preset */
    switch (preset) {
    case RBPF_PREPROCESS_NONE:
        prep->enable_diurnal = 0;
        prep->enable_zero_clamp = 1;
        prep->enable_outlier_clamp = 0;
        prep->zero_floor = RBPF_REAL(1e-8);
        break;
        
    case RBPF_PREPROCESS_EQUITY_US:
        apply_preset_equity_us(prep);
        break;
        
    case RBPF_PREPROCESS_EQUITY_EU:
        apply_preset_equity_eu(prep);
        break;
        
    case RBPF_PREPROCESS_FX_24H:
        apply_preset_fx_24h(prep);
        break;
        
    case RBPF_PREPROCESS_CRYPTO_24H:
        apply_preset_crypto_24h(prep);
        break;
        
    case RBPF_PREPROCESS_FUTURES_US:
        apply_preset_futures_us(prep);
        break;
        
    case RBPF_PREPROCESS_CUSTOM:
        /* User will configure manually */
        prep->enable_diurnal = 0;
        prep->enable_zero_clamp = 1;
        prep->enable_outlier_clamp = 0;
        prep->zero_floor = RBPF_REAL(1e-8);
        break;
    }
    
    /* Initialize running vol to something reasonable */
    prep->running_vol = RBPF_REAL(0.001);  /* 0.1% */
    
    return prep;
}

void rbpf_preprocess_destroy(RBPF_Preprocessor *prep)
{
    free(prep);
}

/*============================================================================
 * DIURNAL PATTERN INITIALIZATION
 *============================================================================*/

void rbpf_preprocess_init_diurnal(
    RBPF_Preprocessor *prep,
    const rbpf_real_t *returns,
    const int *timestamps,
    int n)
{
    if (!prep || !returns || !timestamps || n <= 0) return;
    
    RBPF_DiurnalPattern *d = &prep->diurnal;
    
    /* Reset accumulators */
    memset(d->sum_sq, 0, sizeof(d->sum_sq));
    memset(d->count, 0, sizeof(d->count));
    
    /* Accumulate squared returns per bin */
    for (int i = 0; i < n; i++) {
        int bin = rbpf_preprocess_time_to_bin(prep, timestamps[i]);
        rbpf_real_t r = returns[i];
        d->sum_sq[bin] += (double)(r * r);
        d->count[bin]++;
    }
    
    /* Compute average vol per bin */
    double global_sum = 0;
    int global_count = 0;
    
    for (int b = 0; b < d->n_bins; b++) {
        if (d->count[b] > 0) {
            global_sum += d->sum_sq[b];
            global_count += d->count[b];
        }
    }
    
    double global_avg_sq = (global_count > 0) ? global_sum / global_count : 1.0;
    double global_vol = sqrt(global_avg_sq);
    
    if (global_vol < 1e-10) global_vol = 1e-10;
    
    /* Compute relative factors */
    for (int b = 0; b < d->n_bins; b++) {
        double bin_vol;
        if (d->count[b] >= 10) {  /* Need enough samples */
            bin_vol = sqrt(d->sum_sq[b] / d->count[b]);
        } else {
            bin_vol = global_vol;  /* Fall back to global */
        }
        
        /* Relative factor: bin_vol / global_vol */
        d->vol_factor[b] = (rbpf_real_t)(bin_vol / global_vol);
        
        /* Clamp to reasonable range [0.3, 3.0] */
        if (d->vol_factor[b] < RBPF_REAL(0.3))
            d->vol_factor[b] = RBPF_REAL(0.3);
        if (d->vol_factor[b] > RBPF_REAL(3.0))
            d->vol_factor[b] = RBPF_REAL(3.0);
        
        d->inv_vol_factor[b] = RBPF_REAL(1.0) / d->vol_factor[b];
    }
    
    /* Smooth the pattern (simple moving average) */
    rbpf_real_t smoothed[RBPF_PREPROCESS_MAX_BINS];
    int window = 3;  /* 3-bin smoothing window */
    
    for (int b = 0; b < d->n_bins; b++) {
        double sum = 0;
        int cnt = 0;
        for (int k = -window; k <= window; k++) {
            int idx = b + k;
            if (idx >= 0 && idx < d->n_bins) {
                sum += d->vol_factor[idx];
                cnt++;
            }
        }
        smoothed[b] = (rbpf_real_t)(sum / cnt);
    }
    
    for (int b = 0; b < d->n_bins; b++) {
        d->vol_factor[b] = smoothed[b];
        d->inv_vol_factor[b] = RBPF_REAL(1.0) / smoothed[b];
    }
    
    /* Estimate days seen */
    int ticks_per_day = d->n_bins * 10;  /* Rough estimate */
    d->days_seen = n / ticks_per_day;
    if (d->days_seen < d->warmup_days) {
        d->days_seen = d->warmup_days;  /* Trust the data we have */
    }
}

void rbpf_preprocess_set_diurnal(
    RBPF_Preprocessor *prep,
    const rbpf_real_t *vol_factors,
    int n_bins)
{
    if (!prep || !vol_factors || n_bins <= 0) return;
    
    RBPF_DiurnalPattern *d = &prep->diurnal;
    
    if (n_bins > RBPF_PREPROCESS_MAX_BINS) {
        n_bins = RBPF_PREPROCESS_MAX_BINS;
    }
    d->n_bins = n_bins;
    
    for (int b = 0; b < n_bins; b++) {
        d->vol_factor[b] = vol_factors[b];
        
        /* Clamp */
        if (d->vol_factor[b] < RBPF_REAL(0.3))
            d->vol_factor[b] = RBPF_REAL(0.3);
        if (d->vol_factor[b] > RBPF_REAL(3.0))
            d->vol_factor[b] = RBPF_REAL(3.0);
        
        d->inv_vol_factor[b] = RBPF_REAL(1.0) / d->vol_factor[b];
    }
    
    d->days_seen = d->warmup_days;  /* Trust user-provided data */
}

void rbpf_preprocess_set_default_ushape(RBPF_Preprocessor *prep)
{
    if (!prep) return;
    
    RBPF_DiurnalPattern *d = &prep->diurnal;
    
    /* 
     * Parametric U-shape model for US equities:
     *   vol(t) = 1.0 + A*exp(-B*(t-t0)²) + C*exp(-D*(t1-t)²) - E*(t-tmid)²
     *
     * Where:
     *   t0 = opening time
     *   t1 = closing time  
     *   tmid = midday
     *
     * This gives:
     *   - High vol at open (A term)
     *   - High vol at close (C term)
     *   - Low vol at midday (E term)
     */
    
    int n = d->n_bins;
    if (n <= 0) n = 78;  /* Default US equity */
    d->n_bins = n;
    
    for (int b = 0; b < n; b++) {
        double t = (double)b / (n - 1);  /* Normalized time [0, 1] */
        
        /* Open spike */
        double open_effect = 0.6 * exp(-50.0 * t * t);
        
        /* Close spike */
        double close_effect = 0.5 * exp(-50.0 * (1.0 - t) * (1.0 - t));
        
        /* Midday dip */
        double mid = 0.5;
        double midday_dip = 0.3 * exp(-20.0 * (t - mid) * (t - mid));
        
        /* Combined: base + spikes - dip */
        double factor = 0.8 + open_effect + close_effect - midday_dip;
        
        /* Normalize so average ≈ 1.0 */
        d->vol_factor[b] = (rbpf_real_t)factor;
    }
    
    /* Normalize to mean = 1.0 */
    double sum = 0;
    for (int b = 0; b < n; b++) {
        sum += d->vol_factor[b];
    }
    double mean = sum / n;
    
    for (int b = 0; b < n; b++) {
        d->vol_factor[b] /= (rbpf_real_t)mean;
        d->inv_vol_factor[b] = RBPF_REAL(1.0) / d->vol_factor[b];
    }
    
    d->days_seen = d->warmup_days;  /* Trust default pattern */
}

/*============================================================================
 * ONLINE UPDATE
 *============================================================================*/

void rbpf_preprocess_update(
    RBPF_Preprocessor *prep,
    rbpf_real_t raw_return,
    int time_min)
{
    if (!prep || !prep->enable_online_update) return;
    
    RBPF_DiurnalPattern *d = &prep->diurnal;
    int bin = rbpf_preprocess_time_to_bin(prep, time_min);
    
    /* Update running statistics */
    d->sum_sq[bin] += (double)(raw_return * raw_return);
    d->count[bin]++;
    
    /* EMA update of vol factor */
    if (d->count[bin] >= 10) {
        double new_vol = sqrt(d->sum_sq[bin] / d->count[bin]);
        
        /* Compute global average for normalization */
        double global_sum = 0;
        int global_count = 0;
        for (int b = 0; b < d->n_bins; b++) {
            if (d->count[b] > 0) {
                global_sum += d->sum_sq[b];
                global_count += d->count[b];
            }
        }
        
        if (global_count > 0) {
            double global_vol = sqrt(global_sum / global_count);
            if (global_vol > 1e-10) {
                rbpf_real_t new_factor = (rbpf_real_t)(new_vol / global_vol);
                
                /* Clamp */
                if (new_factor < RBPF_REAL(0.3)) new_factor = RBPF_REAL(0.3);
                if (new_factor > RBPF_REAL(3.0)) new_factor = RBPF_REAL(3.0);
                
                /* EMA blend */
                d->vol_factor[bin] = d->ema_alpha * new_factor +
                                    (RBPF_REAL(1.0) - d->ema_alpha) * d->vol_factor[bin];
                d->inv_vol_factor[bin] = RBPF_REAL(1.0) / d->vol_factor[bin];
            }
        }
    }
}

void rbpf_preprocess_new_day(RBPF_Preprocessor *prep)
{
    if (!prep) return;
    prep->diurnal.days_seen++;
}

rbpf_real_t rbpf_preprocess_get_diurnal_factor(
    const RBPF_Preprocessor *prep,
    int time_min)
{
    if (!prep) return RBPF_REAL(1.0);
    
    int bin = rbpf_preprocess_time_to_bin(prep, time_min);
    return prep->diurnal.vol_factor[bin];
}

/*============================================================================
 * DIAGNOSTICS
 *============================================================================*/

void rbpf_preprocess_print_stats(const RBPF_Preprocessor *prep)
{
    if (!prep) return;
    
    const char *preset_names[] = {
        "NONE", "EQUITY_US", "EQUITY_EU", "FX_24H", "CRYPTO_24H", "FUTURES_US", "CUSTOM"
    };
    
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│               RBPF Preprocessor Statistics                 │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ Preset:              %-20s                │\n", preset_names[prep->preset]);
    printf("│ Diurnal adjustment:  %-5s                                 │\n", 
           prep->enable_diurnal ? "ON" : "OFF");
    printf("│ Zero clamping:       %-5s (floor=%.0e)                  │\n",
           prep->enable_zero_clamp ? "ON" : "OFF", prep->zero_floor);
    printf("│ Outlier clamping:    %-5s (mult=%.1f)                     │\n",
           prep->enable_outlier_clamp ? "ON" : "OFF", prep->outlier_mult);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ Total observations:  %-20llu                │\n", 
           (unsigned long long)prep->n_total);
    printf("│ Zeros clamped:       %-20llu (%.2f%%)          │\n",
           (unsigned long long)prep->n_zeros_clamped,
           prep->n_total > 0 ? 100.0 * prep->n_zeros_clamped / prep->n_total : 0.0);
    printf("│ Outliers clamped:    %-20llu (%.2f%%)          │\n",
           (unsigned long long)prep->n_outliers_clamped,
           prep->n_total > 0 ? 100.0 * prep->n_outliers_clamped / prep->n_total : 0.0);
    printf("│ Running vol:         %-20.6f                │\n", prep->running_vol);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ Diurnal bins:        %-5d                                 │\n", 
           prep->diurnal.n_bins);
    printf("│ Days observed:       %-5d (warmup: %d)                    │\n",
           prep->diurnal.days_seen, prep->diurnal.warmup_days);
    printf("└─────────────────────────────────────────────────────────────┘\n");
}

void rbpf_preprocess_print_diurnal(const RBPF_Preprocessor *prep)
{
    if (!prep) return;
    
    const RBPF_DiurnalPattern *d = &prep->diurnal;
    
    printf("\nDiurnal Pattern (session %02d:%02d - %02d:%02d, %d bins):\n",
           d->session_start_min / 60, d->session_start_min % 60,
           d->session_end_min / 60, d->session_end_min % 60,
           d->n_bins);
    printf("────────────────────────────────────────────────────────────────\n");
    
    /* Find min/max for scaling */
    rbpf_real_t min_f = d->vol_factor[0];
    rbpf_real_t max_f = d->vol_factor[0];
    for (int b = 1; b < d->n_bins; b++) {
        if (d->vol_factor[b] < min_f) min_f = d->vol_factor[b];
        if (d->vol_factor[b] > max_f) max_f = d->vol_factor[b];
    }
    
    /* ASCII bar chart */
    const int bar_width = 40;
    for (int b = 0; b < d->n_bins; b++) {
        int time_min = d->session_start_min + b * d->bin_duration_min;
        int hour = time_min / 60;
        int minute = time_min % 60;
        
        rbpf_real_t f = d->vol_factor[b];
        int bar_len = (int)((f - min_f) / (max_f - min_f + 0.001) * bar_width);
        if (bar_len < 1) bar_len = 1;
        
        printf("%02d:%02d │ %5.2f │", hour, minute, f);
        for (int i = 0; i < bar_len; i++) printf("█");
        printf("\n");
    }
    printf("────────────────────────────────────────────────────────────────\n");
}
