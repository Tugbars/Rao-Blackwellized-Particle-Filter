/**
 * @file test_mmpf_real_data.c
 * @brief Run MMPF on real market data from CSV files
 * 
 * Usage: test_mmpf_real_data <input.csv> <output.csv> [options]
 * 
 * Input CSV format:
 *   timestamp,close,return[,event]
 * 
 * Output CSV format:
 *   timestamp,return,vol,log_vol,vol_std,w_calm,w_trend,w_crisis,
 *   dominant,outlier_frac,ess_min,latency_us
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "mmpf_rocks.h"

/*═══════════════════════════════════════════════════════════════════════════
 * PLATFORM-SPECIFIC TIMING
 *═══════════════════════════════════════════════════════════════════════════*/

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static double get_time_us(void) {
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1e6 / (double)freq.QuadPart;
}
#else
#include <sys/time.h>

static double get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CSV PARSING
 *═══════════════════════════════════════════════════════════════════════════*/

#define MAX_LINE_LEN 1024
#define MAX_ROWS 1000000

typedef struct {
    char timestamp[64];
    double ret;       /* return (not price) */
    char event[64];   /* optional event label */
} DataRow;

typedef struct {
    DataRow *rows;
    int n_rows;
    int capacity;
} Dataset;

static Dataset *dataset_create(int capacity) {
    Dataset *ds = (Dataset *)malloc(sizeof(Dataset));
    ds->rows = (DataRow *)malloc(capacity * sizeof(DataRow));
    ds->n_rows = 0;
    ds->capacity = capacity;
    return ds;
}

static void dataset_destroy(Dataset *ds) {
    if (ds) {
        free(ds->rows);
        free(ds);
    }
}

static int parse_csv_line(char *line, DataRow *row) {
    /* Expected: timestamp,close,return[,event] */
    char *token;
    int col = 0;
    
    row->event[0] = '\0';
    
    token = strtok(line, ",");
    while (token != NULL) {
        /* Trim whitespace and quotes */
        while (*token == ' ' || *token == '"') token++;
        char *end = token + strlen(token) - 1;
        while (end > token && (*end == ' ' || *end == '"' || *end == '\n' || *end == '\r')) {
            *end = '\0';
            end--;
        }
        
        switch (col) {
            case 0:  /* timestamp */
                strncpy(row->timestamp, token, sizeof(row->timestamp) - 1);
                break;
            case 1:  /* close - skip, we use return */
                break;
            case 2:  /* return */
                row->ret = atof(token);
                break;
            case 3:  /* event (optional) */
                strncpy(row->event, token, sizeof(row->event) - 1);
                break;
        }
        col++;
        token = strtok(NULL, ",");
    }
    
    return (col >= 3) ? 0 : -1;  /* Need at least timestamp, close, return */
}

static Dataset *load_csv(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open %s\n", filename);
        return NULL;
    }
    
    Dataset *ds = dataset_create(MAX_ROWS);
    char line[MAX_LINE_LEN];
    int line_num = 0;
    
    /* Skip header */
    if (fgets(line, sizeof(line), fp) == NULL) {
        fclose(fp);
        dataset_destroy(ds);
        return NULL;
    }
    line_num++;
    
    /* Parse data rows */
    while (fgets(line, sizeof(line), fp) != NULL) {
        line_num++;
        
        if (ds->n_rows >= ds->capacity) {
            fprintf(stderr, "WARNING: Exceeded max rows (%d)\n", MAX_ROWS);
            break;
        }
        
        DataRow *row = &ds->rows[ds->n_rows];
        if (parse_csv_line(line, row) == 0) {
            /* Sanity check return */
            if (fabs(row->ret) < 0.5) {  /* Filter out >50% returns as bad data */
                ds->n_rows++;
            }
        }
    }
    
    fclose(fp);
    printf("Loaded %d rows from %s\n", ds->n_rows, filename);
    
    return ds;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MMPF CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    int n_particles;
    int enable_storvik;
    
    /* Hypothesis parameters - realistic for daily data */
    double calm_vol;      /* e.g., 0.008 = 0.8% daily */
    double trend_vol;     /* e.g., 0.015 = 1.5% daily */
    double crisis_vol;    /* e.g., 0.035 = 3.5% daily */
    
} TestConfig;

static TestConfig config_defaults(void) {
    TestConfig cfg;
    cfg.n_particles = 512;
    cfg.enable_storvik = 1;  /* Enable learning for real data */
    
    /* Realistic daily volatility levels for SPY */
    cfg.calm_vol = 0.008;    /* ~0.8% daily (VIX ~12) */
    cfg.trend_vol = 0.015;   /* ~1.5% daily (VIX ~24) */
    cfg.crisis_vol = 0.035;  /* ~3.5% daily (VIX ~55) */
    
    return cfg;
}

static TestConfig config_intraday(void) {
    TestConfig cfg = config_defaults();
    
    /* Scale down for 1-minute bars
     * Daily vol ~1% → 1-min vol ~1%/sqrt(390) ~0.05% */
    cfg.calm_vol = 0.0003;   /* 0.03% per minute */
    cfg.trend_vol = 0.0008;  /* 0.08% per minute */
    cfg.crisis_vol = 0.002;  /* 0.20% per minute */
    
    return cfg;
}

static MMPF_ROCKS *create_mmpf(const TestConfig *cfg) {
    MMPF_Config mmpf_cfg = mmpf_config_defaults();
    mmpf_cfg.n_particles = cfg->n_particles;
    mmpf_cfg.enable_storvik_sync = cfg->enable_storvik;
    
    /* Convert volatility to log-vol (mu_vol = log(sigma)) */
    mmpf_cfg.hypotheses[MMPF_CALM].mu_vol = (rbpf_real_t)log(cfg->calm_vol);
    mmpf_cfg.hypotheses[MMPF_CALM].phi = RBPF_REAL(0.98);
    mmpf_cfg.hypotheses[MMPF_CALM].sigma_eta = RBPF_REAL(0.05);
    
    mmpf_cfg.hypotheses[MMPF_TREND].mu_vol = (rbpf_real_t)log(cfg->trend_vol);
    mmpf_cfg.hypotheses[MMPF_TREND].phi = RBPF_REAL(0.95);
    mmpf_cfg.hypotheses[MMPF_TREND].sigma_eta = RBPF_REAL(0.10);
    
    mmpf_cfg.hypotheses[MMPF_CRISIS].mu_vol = (rbpf_real_t)log(cfg->crisis_vol);
    mmpf_cfg.hypotheses[MMPF_CRISIS].phi = RBPF_REAL(0.85);
    mmpf_cfg.hypotheses[MMPF_CRISIS].sigma_eta = RBPF_REAL(0.20);
    
    /* More responsive for real data */
    mmpf_cfg.base_stickiness = RBPF_REAL(0.95);
    mmpf_cfg.min_stickiness = RBPF_REAL(0.80);
    
    return mmpf_create(&mmpf_cfg);
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN PROCESSING LOOP
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Timing */
    double total_time_us;
    double min_latency_us;
    double max_latency_us;
    double avg_latency_us;
    
    /* Filter health */
    int ess_collapses;      /* ESS < 20 */
    int regime_switches;
    int outlier_detections; /* outlier_frac > 0.5 */
    
    /* Regime distribution */
    int ticks_calm;
    int ticks_trend;
    int ticks_crisis;
    
} RunStats;

static int run_mmpf_on_data(
    MMPF_ROCKS *mmpf,
    const Dataset *ds,
    const char *output_file,
    RunStats *stats)
{
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot create %s\n", output_file);
        return -1;
    }
    
    /* CSV header */
    fprintf(fp, "timestamp,return,vol,log_vol,vol_std,w_calm,w_trend,w_crisis,"
                "dominant,outlier_frac,ess_min,latency_us,event\n");
    
    /* Initialize stats */
    memset(stats, 0, sizeof(RunStats));
    stats->min_latency_us = 1e9;
    
    MMPF_Output out;
    MMPF_Hypothesis prev_dom = MMPF_CALM;
    
    /* Progress tracking */
    int progress_interval = ds->n_rows / 20;
    if (progress_interval < 1) progress_interval = 1;
    
    printf("Processing %d ticks...\n", ds->n_rows);
    
    for (int t = 0; t < ds->n_rows; t++) {
        const DataRow *row = &ds->rows[t];
        
        /* Time the step */
        double t0 = get_time_us();
        mmpf_step(mmpf, (rbpf_real_t)row->ret, &out);
        double t1 = get_time_us();
        double latency = t1 - t0;
        
        /* Update timing stats */
        stats->total_time_us += latency;
        if (latency < stats->min_latency_us) stats->min_latency_us = latency;
        if (latency > stats->max_latency_us) stats->max_latency_us = latency;
        
        /* Get additional outputs */
        rbpf_real_t weights[MMPF_N_MODELS];
        mmpf_get_weights(mmpf, weights);
        
        rbpf_real_t vol = mmpf_get_volatility(mmpf);
        rbpf_real_t log_vol = mmpf_get_log_volatility(mmpf);
        rbpf_real_t vol_std = mmpf_get_volatility_std(mmpf);
        rbpf_real_t outlier = mmpf_get_outlier_fraction(mmpf);
        MMPF_Hypothesis dom = mmpf_get_dominant(mmpf);
        
        /* ESS minimum across models */
        rbpf_real_t ess_min = 1e9;
        for (int k = 0; k < MMPF_N_MODELS; k++) {
            if (mmpf->model_output[k].ess < ess_min) {
                ess_min = mmpf->model_output[k].ess;
            }
        }
        
        /* Update health stats */
        if (ess_min < 20) stats->ess_collapses++;
        if (dom != prev_dom) stats->regime_switches++;
        if (outlier > 0.5) stats->outlier_detections++;
        
        switch (dom) {
            case MMPF_CALM:   stats->ticks_calm++; break;
            case MMPF_TREND:  stats->ticks_trend++; break;
            case MMPF_CRISIS: stats->ticks_crisis++; break;
        }
        
        prev_dom = dom;
        
        /* Write output row */
        fprintf(fp, "%s,%.8f,%.8f,%.6f,%.8f,%.6f,%.6f,%.6f,%d,%.6f,%.2f,%.2f,%s\n",
                row->timestamp,
                row->ret,
                vol,
                log_vol,
                vol_std,
                weights[MMPF_CALM],
                weights[MMPF_TREND],
                weights[MMPF_CRISIS],
                (int)dom,
                outlier,
                ess_min,
                latency,
                row->event);
        
        /* Progress */
        if ((t + 1) % progress_interval == 0) {
            int pct = (t + 1) * 100 / ds->n_rows;
            printf("  %3d%% (%d/%d) - avg latency: %.1f μs\n",
                   pct, t + 1, ds->n_rows, stats->total_time_us / (t + 1));
        }
    }
    
    fclose(fp);
    
    stats->avg_latency_us = stats->total_time_us / ds->n_rows;
    
    return 0;
}

static void print_stats(const RunStats *stats, int n_rows) {
    printf("\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("  MMPF Real Data Results\n");
    printf("════════════════════════════════════════════════════════════\n\n");
    
    printf("  Timing:\n");
    printf("    Total:     %.2f ms\n", stats->total_time_us / 1000.0);
    printf("    Avg:       %.2f μs/tick\n", stats->avg_latency_us);
    printf("    Min:       %.2f μs\n", stats->min_latency_us);
    printf("    Max:       %.2f μs\n", stats->max_latency_us);
    printf("    Throughput: %.0f ticks/sec\n", 1e6 / stats->avg_latency_us);
    printf("\n");
    
    printf("  Filter Health:\n");
    printf("    ESS collapses:   %d (%.2f%%)\n", 
           stats->ess_collapses, 100.0 * stats->ess_collapses / n_rows);
    printf("    Regime switches: %d\n", stats->regime_switches);
    printf("    Outliers (>50%%): %d (%.2f%%)\n",
           stats->outlier_detections, 100.0 * stats->outlier_detections / n_rows);
    printf("\n");
    
    printf("  Regime Distribution:\n");
    printf("    Calm:   %5d ticks (%5.1f%%)\n", 
           stats->ticks_calm, 100.0 * stats->ticks_calm / n_rows);
    printf("    Trend:  %5d ticks (%5.1f%%)\n",
           stats->ticks_trend, 100.0 * stats->ticks_trend / n_rows);
    printf("    Crisis: %5d ticks (%5.1f%%)\n",
           stats->ticks_crisis, 100.0 * stats->ticks_crisis / n_rows);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

static void print_usage(const char *prog) {
    printf("Usage: %s <input.csv> <output.csv> [options]\n\n", prog);
    printf("Options:\n");
    printf("  --particles N    Number of particles (default: 512)\n");
    printf("  --no-learning    Disable Storvik parameter learning\n");
    printf("  --intraday       Use intraday volatility scaling\n");
    printf("  --help           Show this help\n");
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *output_file = argv[2];
    
    /* Parse options */
    TestConfig cfg = config_defaults();
    int is_intraday = 0;
    
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--particles") == 0 && i + 1 < argc) {
            cfg.n_particles = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-learning") == 0) {
            cfg.enable_storvik = 0;
        } else if (strcmp(argv[i], "--intraday") == 0) {
            is_intraday = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (is_intraday) {
        cfg = config_intraday();
    }
    
    printf("\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("  MMPF Real Data Integration Test\n");
    printf("════════════════════════════════════════════════════════════\n\n");
    printf("  Input:      %s\n", input_file);
    printf("  Output:     %s\n", output_file);
    printf("  Particles:  %d\n", cfg.n_particles);
    printf("  Learning:   %s\n", cfg.enable_storvik ? "ON" : "OFF");
    printf("  Mode:       %s\n", is_intraday ? "Intraday" : "Daily");
    printf("\n");
    printf("  Hypothesis Volatilities:\n");
    printf("    Calm:   %.3f%% (log: %.2f)\n", cfg.calm_vol * 100, log(cfg.calm_vol));
    printf("    Trend:  %.3f%% (log: %.2f)\n", cfg.trend_vol * 100, log(cfg.trend_vol));
    printf("    Crisis: %.3f%% (log: %.2f)\n", cfg.crisis_vol * 100, log(cfg.crisis_vol));
    printf("\n");
    
    /* Load data */
    Dataset *ds = load_csv(input_file);
    if (!ds || ds->n_rows == 0) {
        fprintf(stderr, "ERROR: No data loaded\n");
        return 1;
    }
    
    /* Create MMPF */
    MMPF_ROCKS *mmpf = create_mmpf(&cfg);
    if (!mmpf) {
        fprintf(stderr, "ERROR: Failed to create MMPF\n");
        dataset_destroy(ds);
        return 1;
    }
    
    /* Initialize with first return's implied volatility */
    double init_vol = fabs(ds->rows[0].ret);
    if (init_vol < 0.001) init_vol = cfg.calm_vol;
    mmpf_reset(mmpf, (rbpf_real_t)init_vol);
    
    /* Run */
    RunStats stats;
    int rc = run_mmpf_on_data(mmpf, ds, output_file, &stats);
    
    if (rc == 0) {
        print_stats(&stats, ds->n_rows);
        printf("Output saved to: %s\n\n", output_file);
    }
    
    /* Cleanup */
    mmpf_destroy(mmpf);
    dataset_destroy(ds);
    
    return rc;
}
