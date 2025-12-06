# RBPF-KSC: Rao-Blackwellized Particle Filter for Stochastic Volatility

## Complete Development Documentation

**Version:** 1.0  
**Date:** December 2024  
**Author:** Development Log  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [Development Timeline](#3-development-timeline)
4. [Architecture & Design](#4-architecture--design)
5. [Core Algorithms](#5-core-algorithms)
6. [Optimizations](#6-optimizations)
7. [Bug Fixes & Lessons Learned](#7-bug-fixes--lessons-learned)
8. [Final Performance](#8-final-performance)
9. [API Reference](#9-api-reference)
10. [Usage Examples](#10-usage-examples)
11. [Future Enhancements](#11-future-enhancements)

---

## 1. Executive Summary

### Project Goal

Develop a high-frequency trading (HFT) volatility filter that:
- Estimates latent log-volatility from noisy price returns
- Classifies market regimes (calm → crisis) in real-time
- Learns model parameters online without offline calibration
- Operates within a 200μs per-tick latency budget

### Final Achievements

| Metric | Target | Achieved |
|--------|--------|----------|
| Latency (200 particles) | < 50μs | **10.1μs** ✓ |
| MAE (volatility) | < 0.05 | **0.0158** ✓ |
| Tail MAE (crisis periods) | < 0.15 | **0.0678** ✓ |
| Regime accuracy | > 60% | **69.1%** ✓ |
| Correlation | > 0.85 | **0.893** ✓ |
| Liu-West learning error | < 3.0 | **1.73** ✓ |

### Technology Stack

- **Language:** C11 with Intel MKL
- **Precision:** Configurable float/double via `RBPF_USE_DOUBLE`
- **SIMD:** AVX-512/AVX2 via MKL vectorized math
- **RNG:** PCG32 (particles) + MKL VSL (batch ICDF resampling)
- **Compiler:** GCC/Clang/MSVC compatible

---

## 2. Theoretical Foundation

### 2.1 Stochastic Volatility Model

The underlying model is an Ornstein-Uhlenbeck process for log-volatility:

```
State equation:  ss   ℓ_t = (1-θ)ℓ_{t-1} + θμ + η_t,    η_t ~ N(0, σ²_vol)
Observation:        y_t = exp(ℓ_t/2) × ε_t,           ε_t ~ N(0, 1)
```

Where:
- `ℓ_t` = log-volatility (state variable)
- `y_t` = observed return
- `θ` = mean reversion speed (0 = random walk, 1 = white noise)
- `μ` = long-run mean log-volatility
- `σ_vol` = volatility of volatility

### 2.2 The Non-Gaussian Challenge

Squaring the observation equation:
```
log(y²_t) = ℓ_t + log(ε²_t)
```

The noise term `log(ε²_t)` follows a log-chi-squared distribution, which is:
- Highly skewed (long left tail)
- NOT Gaussian

This breaks the standard Kalman filter assumption.

### 2.3 Kim-Shephard-Chib (1998) → Omori (2007) Mixture Approximation

**Solution:** Approximate `log(χ²(1))` as a Gaussian mixture:

```
p(log(ε²)) ≈ Σ_k π_k × N(m_k, v²_k)
```

**Evolution:**

| Version | Components | Tail Accuracy | Reference |
|---------|------------|---------------|-----------|
| KSC 1998 | 7 | Good | Kim, Shephard, Chib (1998) |
| Omori 2007 | 10 | Excellent | Omori, Chib, Shephard, Nakajima (2007) |

We use **Omori 10-component** for better crisis detection:

| k | π_k | m_k | v²_k |
|---|-----|-----|------|
| 0 | 0.00609 | 1.92677 | 0.11265 |
| 1 | 0.04775 | 1.34744 | 0.17788 |
| 2 | 0.13057 | 0.73504 | 0.26768 |
| 3 | 0.20674 | 0.02266 | 0.40611 |
| 4 | 0.22715 | -0.85173 | 0.62699 |
| 5 | 0.18842 | -1.97278 | 0.98583 |
| 6 | 0.12047 | -3.46788 | 1.57469 |
| 7 | 0.05591 | -5.55246 | 2.54498 |
| 8 | 0.01575 | -8.68384 | 4.16591 |
| 9 | 0.00115 | -14.65000 | 7.33342 |

### 2.4 Rao-Blackwellization

**Key insight:** Given the mixture component, the model is linear-Gaussian.

**Strategy:**
1. Sample discrete variables (regime, mixture component) with particles
2. Integrate continuous state (log-vol) analytically via Kalman filter
3. Each particle carries a Kalman filter, not a point estimate

**Variance reduction:** O(√N) improvement over standard particle filter.

### 2.5 Regime-Switching Extension

Extend to multiple volatility regimes with Markov transitions:

```
Regime r ∈ {0, 1, 2, 3}  (Calm → Medium → High → Crisis)
P(r_t | r_{t-1}) = Transition matrix Π
Each regime has: (θ_r, μ_r, σ_r)
```

### 2.6 State Definition: H=2 (Log-Volatility)

**Critical assumption:** Our state is log-volatility, NOT log-variance:

```
ℓ_t = log(σ_t)           ← Our state (H=2 in observation)
h_t = log(σ²_t) = 2ℓ_t   ← Some papers use this (H=1)
```

**Parameter conversion from H=1 literature:**
```
μ_vol (here)  = μ_h / 2
σ_vol (here)  = σ_h / 2
θ (here)      = θ_h  (unchanged)
```

---

## 3. Development Timeline

### Phase 1: Self-Aware Detection (Foundation)

**Goal:** Build detection signals without external models.

**Deliverables:**
- Surprise signal: `-log(p(y_t | y_{1:t-1}))`
- Regime entropy: `-Σ p(r) log(p(r))`
- Vol ratio: `EMA_short / EMA_long`
- Regime change flags

**Key insight:** The Kalman filter provides exact marginal likelihood for free.

### Phase 2: RBPF-KSC Core

**Goal:** Implement the particle filter with KSC mixture.

**Deliverables:**
- Particle state: `(μ, var, regime)` per particle
- Predict step: OU dynamics with Kalman propagation
- Update step: 7-component (later 10) mixture Kalman update
- Systematic resampling with ESS-based triggering

**Initial bugs:**
- Variance collapse (fixed with law of total variance)
- Numerical underflow (fixed with log-sum-exp)

### Phase 3: Liu-West Parameter Learning

**Goal:** Learn regime parameters online without PMMH.

**Deliverables:**
- Per-particle parameter storage: `μ_vol[i][r]`, `σ_vol[i][r]`
- Liu-West shrinkage kernel
- Adaptive resampling for learning mode

**Key insight:** Liu-West only learns during resampling!

### Phase 4: Float/Double Precision Switch

**Goal:** Support both float (HFT) and double (validation).

**Deliverables:**
- `RBPF_USE_DOUBLE` compile flag
- `rbpf_real_t` type abstraction
- MKL function wrappers for both precisions

### Phase 5: HFT Optimizations

**Goal:** Minimize latency for live trading.

**Deliverables:**
- Log-sum-exp numerical stability
- Fused resample-copy (pointer swap, no memcpy)
- MKL batch ICDF resampling
- Scalar loops for small N (VML overhead)
- Precomputed transition LUT

### Phase 6: Accuracy Improvements

**Goal:** Fix systematic biases and improve convergence.

**Deliverables:**
- Omori 10-component mixture (replaces KSC 7)
- Order constraint (prevents label switching)
- Regime repulsion (prevents data theft)
- Hysteresis smoothing (reduces flickering)

---

## 4. Architecture & Design

### 4.1 Data Layout: Structure of Arrays (SoA)

```c
/* Particle state - SoA for SIMD efficiency */
rbpf_real_t *mu;           /* [n] log-vol mean */
rbpf_real_t *var;          /* [n] log-vol variance */
int *regime;               /* [n] regime index */
rbpf_real_t *log_weight;   /* [n] log-weights */

/* Double buffers for pointer-swap resampling */
rbpf_real_t *mu_tmp;
rbpf_real_t *var_tmp;
int *regime_tmp;
```

### 4.2 Memory Alignment

All arrays aligned to 64 bytes for cache line optimization:

```c
#define RBPF_ALIGN 64
rbpf_real_t *mu = mkl_malloc(n * sizeof(rbpf_real_t), RBPF_ALIGN);
```

### 4.3 Zero-Allocation Hot Path

All workspace preallocated at creation:

```c
/* Workspace - NO malloc in step() */
rbpf_real_t *mu_pred;      /* [n] predicted mean */
rbpf_real_t *var_pred;     /* [n] predicted variance */
rbpf_real_t *lik_total;    /* [n] total likelihood per particle */
rbpf_real_t *log_lik_buffer; /* [K*n] per-component likelihoods */
/* ... etc ... */
```

### 4.4 Liu-West Parameter Storage

```c
/* Per-particle, per-regime parameters */
rbpf_real_t *particle_mu_vol;      /* [n × n_regimes] */
rbpf_real_t *particle_sigma_vol;   /* [n × n_regimes] */

/* Sufficient statistics for shrinkage */
rbpf_real_t lw_mu_vol_mean[RBPF_MAX_REGIMES];
rbpf_real_t lw_mu_vol_var[RBPF_MAX_REGIMES];
```

---

## 5. Core Algorithms

### 5.1 Predict Step

```
For each particle i:
    r = regime[i]
    θ = params[r].theta
    μ_vol = particle_mu_vol[i][r]  (or global if not learning)
    q = σ²_vol
    
    # Kalman predict
    μ_pred[i] = (1-θ) × μ[i] + θ × μ_vol
    P_pred[i] = (1-θ)² × P[i] + q
    
    # Regime transition (LUT-based)
    u = uniform_random()
    regime[i] = trans_lut[r][floor(u × 1024)]
```

### 5.2 Update Step (10-Component Mixture Kalman)

```
y = log(return²)  # Observation
H = 2.0           # Observation matrix (log-vol to log-var)

For each particle i:
    # Log-sum-exp for numerical stability
    max_ll = -∞
    
    For each component k = 0..9:
        m_k, v²_k, π_k = Omori_params[k]
        
        # Kalman innovation
        S_k = H² × P_pred[i] + v²_k
        K_k = H × P_pred[i] / S_k
        innov = y - H × μ_pred[i] - m_k
        
        # Log-likelihood
        log_lik[k] = log(π_k) - 0.5×log(S_k) - 0.5×innov²/S_k
        max_ll = max(max_ll, log_lik[k])
    
    # Numerical stable sum
    lik_sum = Σ_k exp(log_lik[k] - max_ll)
    log_weight[i] += max_ll + log(lik_sum)
    
    # Collapse mixture to single Gaussian (GPB1)
    # Using LAW OF TOTAL VARIANCE: Var[X] = E[X²] - E[X]²
    μ[i] = Σ_k w_k × μ_k
    E_X2 = Σ_k w_k × (P_k + μ_k²)
    P[i] = E_X2 - μ[i]²
```

### 5.3 Resampling (Systematic with ESS Trigger)

```
# Compute ESS
w_norm = softmax(log_weight)
ESS = 1 / Σ w²_norm

# Adaptive threshold
if liu_west_enabled:
    threshold = 0.8 × n  # Aggressive for learning
    if ticks_since_resample > max_ticks:
        force_resample = true
else:
    threshold = 0.5 × n  # Standard

if ESS < threshold or force_resample:
    # Systematic resampling
    cumsum = cumulative_sum(w_norm)
    u₀ = uniform(0, 1/n)
    
    for i = 0..n-1:
        u = u₀ + i/n
        j = binary_search(cumsum, u)
        
        # Fused copy (keep source hot in cache)
        mu_tmp[i] = mu[j]
        var_tmp[i] = var[j]
        regime_tmp[i] = regime[j]
        indices[i] = j
    
    # Pointer swap (no memcpy!)
    swap(mu, mu_tmp)
    swap(var, var_tmp)
    swap(regime, regime_tmp)
    
    # Reset weights
    log_weight[:] = 0
    
    # Liu-West update
    liu_west_resample(indices)
    
    # Regime diversity enforcement
    enforce_min_particles_per_regime()
```

### 5.4 Liu-West Parameter Learning

```
# Only runs during resampling, after warmup

a = shrinkage  # 0.92 (closer to 1 = slower adaptation)
h = sqrt(1 - a²)

For each particle i:
    parent = indices[i]
    r = regime[i]  # Current regime
    
    # Only update parameters for CURRENT regime
    # (other regimes just copy from parent)
    
    μ_mean = lw_mu_vol_mean[r]  # Weighted mean of particles in regime r
    μ_var = lw_mu_vol_var[r]    # Variance of particles in regime r
    
    # Liu-West kernel
    μ_new = a × μ_parent + (1-a) × μ_mean + h × sqrt(μ_var) × randn()
    
    # Clamp to bounds
    μ_new = clamp(μ_new, min_mu_vol, max_mu_vol)
    
    particle_mu_vol[i][r] = μ_new

# ORDER CONSTRAINT: Enforce μ₀ < μ₁ < μ₂ < μ₃
For each particle i:
    sort(particle_mu_vol[i][:])
    
    # MINIMUM SEPARATION: Prevent data theft
    for r = 1..n_regimes-1:
        if μ[r] - μ[r-1] < min_sep:
            μ[r] = μ[r-1] + min_sep
```

### 5.5 Regime Smoothing (Hysteresis)

```
# Prevents flickering between adjacent regimes

dominant = argmax(regime_probs)
max_prob = max(regime_probs)

if dominant == stable_regime:
    # Same as current - reset candidate
    candidate = dominant
    hold_count = 0
    
elif dominant == candidate:
    # Same as candidate - increment hold
    hold_count++
    
    if hold_count >= hold_threshold OR max_prob >= prob_threshold:
        stable_regime = dominant
        hold_count = 0
        
else:
    # New candidate
    candidate = dominant
    hold_count = 1
    
    if max_prob >= prob_threshold:
        # Immediate switch for high confidence
        stable_regime = dominant
        hold_count = 0

output.smoothed_regime = stable_regime
```

---

## 6. Optimizations

### 6.1 Log-Sum-Exp Stability

**Problem:** Direct likelihood multiplication → underflow to zero.

**Solution:**
```c
/* Instead of: lik = Π_k exp(log_lik[k]) */

/* Find max for stability */
max_ll = max(log_lik[0..K-1]);

/* Sum exp of differences */
sum = Σ_k exp(log_lik[k] - max_ll);

/* Total log-likelihood */
log_weight += max_ll + log(sum);
```

### 6.2 Fused Resample-Copy

**Problem:** Separate resampling index generation + copy = 2 passes.

**Solution:**
```c
/* Single pass: generate index and copy immediately */
for (int i = 0; i < n; i++) {
    while (cumsum[j] < u) j++;
    
    /* Copy immediately - keeps mu[j] hot in cache */
    mu_tmp[i] = mu[j];
    var_tmp[i] = var[j];
    regime_tmp[i] = regime[j];
    indices[i] = j;  /* Still needed for Liu-West */
}

/* Pointer swap - O(1) instead of O(n) memcpy */
swap(mu, mu_tmp);
```

### 6.3 Transition LUT

**Problem:** Computing regime transitions requires sampling + CDF inversion.

**Solution:** Precompute 1024-entry lookup table per regime:

```c
/* Build at initialization */
for (int r = 0; r < n_regimes; r++) {
    for (int i = 0; i < 1024; i++) {
        u = i / 1024.0;
        trans_lut[r][i] = inverse_cdf(cumsum[r], u);
    }
}

/* Use in hot path: O(1) lookup */
int next_regime = trans_lut[current_regime][rand() & 1023];
```

### 6.4 Scalar vs VML Decision

**Problem:** MKL VML has ~500ns startup overhead.

**Solution:**
```c
if (n < 500) {
    /* Scalar loop - avoid VML overhead */
    for (int i = 0; i < n; i++) {
        result[i] = expf(input[i]);
    }
} else {
    /* VML - amortized overhead */
    vsExp(n, input, result);
}
```

### 6.5 MSVC Portability

**Problem:** MSVC requires `/openmp:experimental` for `#pragma omp simd`.

**Solution:**
```c
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    #define RBPF_PRAGMA_SIMD  /* Rely on auto-vectorization */
#else
    #define RBPF_PRAGMA_SIMD _Pragma("omp simd")
#endif
```

---

## 7. Bug Fixes & Lessons Learned

### 7.1 Variance Collapse (Law of Total Variance)

**Symptom:** Filter overconfident, variance too small.

**Root cause:** GPB1 collapse used only `E[Var[X|K]]`, missing `Var[E[X|K]]`.

**Fix:**
```c
/* WRONG: Only within-component variance */
var = Σ_k w_k × var_k;

/* CORRECT: Law of total variance */
E_X2 = Σ_k w_k × (var_k + mu_k²);
var = E_X2 - mean²;
```

**Lesson:** When collapsing a mixture, always use `E[X²] - E[X]²`.

### 7.2 Label Switching

**Symptom:** Regime 1 learned crisis params, regime 3 learned calm params.

**Root cause:** Unconstrained mixture models can swap labels.

**Fix:** Order constraint + minimum separation:
```c
/* After Liu-West update, enforce μ₀ < μ₁ < μ₂ < μ₃ */
sort(particle_mu_vol[i][:]);

/* Plus minimum gap */
for (r = 1; r < n_regimes; r++) {
    if (mu[r] - mu[r-1] < 0.5) {
        mu[r] = mu[r-1] + 0.5;
    }
}
```

**Lesson:** Mixture models need identifiability constraints.

### 7.3 Data Theft (Regime Starvation)

**Symptom:** High-vol regime stuck at low value, wouldn't converge.

**Root cause:** Adjacent regimes clustered together. Lower regime (with more particles) "stole" observations from upper regime.

**Diagnosis:**
```
R2 at -3.6, R3 at -3.1 (both in "medium" zone)
When crisis data arrives (-1.6):
  - R3 should capture it, but is far away
  - R2 has more particles, steals likelihood
  - R3 starves, can't learn to rise
  - R2 wants to rise, but R3 is the ceiling
```

**Fix:** "Jaws of Life" repulsion:
```c
/* Push regimes apart to prevent data theft */
const float min_separation = 0.6;

for (r = 0; r < n_regimes - 1; r++) {
    gap = mean[r+1] - mean[r];
    if (gap < min_separation) {
        mean[r+1] = mean[r] + min_separation;
        var[r+1] += 0.05;  /* Encourage exploration */
    }
}
```

**Lesson:** In ordered mixtures, upper components need "room to breathe".

### 7.4 Liu-West Regime Contamination

**Symptom:** All regime parameters drifted to same value.

**Root cause:** `compute_stats` weighted ALL particles for each regime, regardless of which regime they were in.

**Fix:** Only count particles currently IN that regime:
```c
for (int i = 0; i < n; i++) {
    if (regime[i] != r) continue;  /* KEY FIX */
    
    sum_mu += w[i] * particle_mu_vol[i][r];
    sum_w += w[i];
}
mean[r] = sum_mu / sum_w;
```

**Lesson:** Per-regime statistics must use per-regime filtering.

### 7.5 High ESS Prevents Learning

**Symptom:** Liu-West params never moved despite good tracking.

**Root cause:** ESS stayed at 70%+, so resampling rarely triggered. Liu-West only updates during resampling.

**Fix:** Aggressive resampling for learning mode:
```c
if (liu_west_enabled) {
    threshold = 0.8 * n;  /* Resample at 80% ESS */
    max_ticks = 5;        /* Force every 5 ticks */
} else {
    threshold = 0.5 * n;  /* Standard 50% */
}
```

**Lesson:** Online learning requires frequent resampling, even when ESS is healthy.

### 7.6 Regime Flickering

**Symptom:** Regime estimate oscillated rapidly between 1-2-3.

**Root cause:** Instantaneous particle distribution is noisy.

**Fix:** Hysteresis smoothing:
```c
/* Require 8 consecutive ticks OR 75% probability to switch */
if (hold_count >= 8 || max_prob >= 0.75) {
    stable_regime = candidate;
}
```

**Lesson:** Temporal smoothing is essential for regime classification.

---

## 8. Final Performance

### 8.1 Latency Benchmark

| Particles | Mean (μs) | P50 (μs) | P99 (μs) | Max (μs) |
|-----------|-----------|----------|----------|----------|
| 50 | 5.3 | 4.7 | 16.1 | 25.2 |
| 100 | 7.8 | 4.3 | 8.0 | 11.6 |
| **200** | **10.9** | **7.9** | **23.9** | **240.2** |
| 500 | 20.9 | 19.6 | 40.9 | 178.6 |
| 1000 | 41.4 | 38.6 | 76.6 | 135.5 |
| 2000 | 82.9 | 76.7 | 207.4 | 770.8 |

### 8.2 Accuracy Metrics (n=200)

| Metric | Value |
|--------|-------|
| MAE (volatility) | 0.0158 |
| RMSE (volatility) | 0.0344 |
| MAE (log-vol) | 0.2357 |
| Tail MAE (90th pct) | 0.0678 |
| Max error | 0.5511 |
| Correlation | 0.893 |
| Regime accuracy | 69.1% |

### 8.3 Liu-West Learning (5000 obs)

| Regime | Initial | Learned | True | Error |
|--------|---------|---------|------|-------|
| 0 (Calm) | -5.30 | -5.09 | -4.61 | 0.48 |
| 1 (Medium) | -4.31 | -4.02 | -3.51 | 0.52 |
| 2 (High) | -3.59 | -2.74 | -2.53 | 0.21 |
| 3 (Crisis) | -2.92 | -2.13 | -1.61 | 0.52 |
| **Total** | | | | **1.73** |

### 8.4 Latency Budget

```
Component       Time      Budget    Status
─────────────────────────────────────────
SSA             140.0 μs  140 μs    ✓
RBPF-KSC        10.1 μs   50 μs     ✓
Kelly           0.5 μs    10 μs     ✓
─────────────────────────────────────────
Total           150.6 μs  200 μs    ✓
Headroom        49.4 μs             
```

---

## 9. API Reference

### 9.1 Core Functions

```c
/* Create/Destroy */
RBPF_KSC* rbpf_ksc_create(int n_particles, int n_regimes);
void rbpf_ksc_destroy(RBPF_KSC *rbpf);

/* Configuration */
void rbpf_ksc_set_regime_params(RBPF_KSC *rbpf, int r,
                                 rbpf_real_t theta,      /* Mean reversion [0,1] */
                                 rbpf_real_t mu_vol,     /* Long-run mean (log scale) */
                                 rbpf_real_t sigma_vol); /* Vol-of-vol */

void rbpf_ksc_build_transition_lut(RBPF_KSC *rbpf, 
                                    const rbpf_real_t *trans_matrix);

void rbpf_ksc_set_regularization(RBPF_KSC *rbpf,
                                  rbpf_real_t h_mu,   /* State jitter */
                                  rbpf_real_t h_var); /* Variance jitter */

/* Initialization */
void rbpf_ksc_init(RBPF_KSC *rbpf, 
                    rbpf_real_t mu0,   /* Initial log-vol mean */
                    rbpf_real_t var0); /* Initial log-vol variance */

void rbpf_ksc_warmup(RBPF_KSC *rbpf);  /* JIT warm-up */

/* Main step function */
void rbpf_ksc_step(RBPF_KSC *rbpf, 
                    rbpf_real_t y,        /* Observed return */
                    RBPF_KSC_Output *out); /* Output struct */
```

### 9.2 Regime Configuration

```c
/* Regime diversity (prevent particle collapse) */
void rbpf_ksc_set_regime_diversity(RBPF_KSC *rbpf,
                                    int min_per_regime,      /* Min particles per regime */
                                    rbpf_real_t mutation_prob); /* Random mutation rate */

/* Regime smoothing (prevent flickering) */
void rbpf_ksc_set_regime_smoothing(RBPF_KSC *rbpf,
                                    int hold_threshold,      /* Ticks before switch */
                                    rbpf_real_t prob_threshold); /* Prob for immediate */
```

### 9.3 Liu-West Parameter Learning

```c
/* Enable/disable */
void rbpf_ksc_enable_liu_west(RBPF_KSC *rbpf,
                               rbpf_real_t shrinkage,  /* 0.9-0.98 */
                               int warmup_ticks);      /* Ticks before learning */

void rbpf_ksc_disable_liu_west(RBPF_KSC *rbpf);

/* Configuration */
void rbpf_ksc_set_liu_west_bounds(RBPF_KSC *rbpf,
                                   rbpf_real_t min_mu_vol,
                                   rbpf_real_t max_mu_vol,
                                   rbpf_real_t min_sigma_vol,
                                   rbpf_real_t max_sigma_vol);

void rbpf_ksc_set_liu_west_resample(RBPF_KSC *rbpf,
                                     rbpf_real_t ess_threshold,
                                     int max_ticks_no_resample);

/* Query learned parameters */
void rbpf_ksc_get_learned_params(const RBPF_KSC *rbpf, int regime,
                                  rbpf_real_t *mu_vol_out,
                                  rbpf_real_t *sigma_vol_out);

/* Inject offline PMMH results */
void rbpf_ksc_inject_pmmh(RBPF_KSC *rbpf, int regime,
                           rbpf_real_t pmmh_mu_vol,
                           rbpf_real_t pmmh_sigma_vol,
                           rbpf_real_t blend);  /* 0=keep, 1=full reset */
```

### 9.4 Output Structure

```c
typedef struct {
    /* State estimates */
    rbpf_real_t vol_mean;        /* E[exp(ℓ)] */
    rbpf_real_t log_vol_mean;    /* E[ℓ] */
    rbpf_real_t log_vol_var;     /* Var[ℓ] */
    rbpf_real_t ess;             /* Effective sample size */
    
    /* Regime */
    rbpf_real_t regime_probs[RBPF_MAX_REGIMES];
    int dominant_regime;         /* Instantaneous */
    int smoothed_regime;         /* With hysteresis */
    
    /* Self-aware signals */
    rbpf_real_t marginal_lik;    /* p(y_t | y_{1:t-1}) */
    rbpf_real_t surprise;        /* -log(marginal_lik) */
    rbpf_real_t vol_ratio;       /* EMA_short / EMA_long */
    rbpf_real_t regime_entropy;  /* -Σ p·log(p) */
    
    /* Detection flags */
    int regime_changed;          /* 0 or 1 */
    int change_type;             /* 0=none, 1=structural, 2=vol_shock, 3=surprise */
    
    /* Learned parameters (if Liu-West enabled) */
    rbpf_real_t learned_mu_vol[RBPF_MAX_REGIMES];
    rbpf_real_t learned_sigma_vol[RBPF_MAX_REGIMES];
    
    /* Diagnostics */
    int resampled;
} RBPF_KSC_Output;
```

---

## 10. Usage Examples

### 10.1 Basic Usage

```c
#include "rbpf_ksc.h"

int main() {
    /* Create filter */
    RBPF_KSC *rbpf = rbpf_ksc_create(200, 4);  /* 200 particles, 4 regimes */
    
    /* Configure regimes */
    rbpf_ksc_set_regime_params(rbpf, 0, 0.05f, logf(0.01f), 0.05f);  /* Calm */
    rbpf_ksc_set_regime_params(rbpf, 1, 0.08f, logf(0.03f), 0.10f);  /* Medium */
    rbpf_ksc_set_regime_params(rbpf, 2, 0.12f, logf(0.08f), 0.20f);  /* High */
    rbpf_ksc_set_regime_params(rbpf, 3, 0.15f, logf(0.20f), 0.30f);  /* Crisis */
    
    /* Transition matrix (row-major) */
    float trans[16] = {
        0.92f, 0.05f, 0.02f, 0.01f,
        0.05f, 0.88f, 0.05f, 0.02f,
        0.02f, 0.05f, 0.88f, 0.05f,
        0.01f, 0.02f, 0.05f, 0.92f
    };
    rbpf_ksc_build_transition_lut(rbpf, trans);
    
    /* Initialize */
    rbpf_ksc_init(rbpf, logf(0.01f), 0.1f);
    rbpf_ksc_warmup(rbpf);
    
    /* Process tick */
    RBPF_KSC_Output out;
    float return_t = 0.02f;  /* 2% return */
    
    rbpf_ksc_step(rbpf, return_t, &out);
    
    printf("Volatility: %.4f\n", out.vol_mean);
    printf("Regime: %d\n", out.smoothed_regime);
    printf("Surprise: %.2f\n", out.surprise);
    
    rbpf_ksc_destroy(rbpf);
    return 0;
}
```

### 10.2 With Liu-West Learning

```c
RBPF_KSC *rbpf = rbpf_ksc_create(200, 4);

/* Set initial (possibly wrong) params */
rbpf_ksc_set_regime_params(rbpf, 0, 0.05f, logf(0.015f), 0.05f);
rbpf_ksc_set_regime_params(rbpf, 1, 0.08f, logf(0.025f), 0.10f);
rbpf_ksc_set_regime_params(rbpf, 2, 0.15f, logf(0.060f), 0.20f);
rbpf_ksc_set_regime_params(rbpf, 3, 0.20f, logf(0.150f), 0.30f);

/* Enable Liu-West */
rbpf_ksc_enable_liu_west(rbpf, 0.92f, 30);  /* shrinkage=0.92, warmup=30 */
rbpf_ksc_set_liu_west_resample(rbpf, 0.80f, 5);  /* ESS<80%, force every 5 */
rbpf_ksc_set_liu_west_bounds(rbpf,
    logf(0.001f), logf(0.4f),   /* μ_vol bounds */
    0.01f, 0.6f);                /* σ_vol bounds */

/* Configure regime diversity */
rbpf_ksc_set_regime_diversity(rbpf, 8, 0.01f);  /* 8 min per regime, 1% mutation */
rbpf_ksc_set_regime_smoothing(rbpf, 8, 0.75f);  /* 8 ticks OR 75% prob */

rbpf_ksc_init(rbpf, logf(0.01f), 0.1f);

/* Process data */
for (int t = 0; t < n_obs; t++) {
    rbpf_ksc_step(rbpf, returns[t], &out);
    
    /* Every 100 ticks, print learned params */
    if (t % 100 == 0) {
        printf("t=%d: μ_vol = [%.3f, %.3f, %.3f, %.3f]\n", t,
               out.learned_mu_vol[0], out.learned_mu_vol[1],
               out.learned_mu_vol[2], out.learned_mu_vol[3]);
    }
}
```

### 10.3 Integration with Trading System

```c
/* Real-time trading loop */
void on_tick(float mid_price, RBPF_KSC *rbpf, KellyState *kelly) {
    static float prev_price = 0;
    
    if (prev_price > 0) {
        /* Compute return */
        float ret = (mid_price - prev_price) / prev_price;
        
        /* Update volatility filter */
        RBPF_KSC_Output vol_out;
        rbpf_ksc_step(rbpf, ret, &vol_out);
        
        /* Risk management */
        if (vol_out.smoothed_regime >= 2) {
            /* High/Crisis regime - reduce position */
            kelly->max_leverage = 0.5f;
        } else {
            kelly->max_leverage = 2.0f;
        }
        
        /* Surprise-based alerts */
        if (vol_out.surprise > 5.0f) {
            trigger_alert("High surprise event detected");
        }
        
        /* Update Kelly sizing */
        kelly_update(kelly, vol_out.vol_mean, vol_out.log_vol_var);
    }
    
    prev_price = mid_price;
}
```

---

## 11. Future Enhancements

### 11.1 Production Roadmap (HFT-Optimized)

The academic progression (GPB1 → GPB2 → IMM) is **not optimal for HFT**. Here's the corrected roadmap based on practical value vs latency cost:

#### Stage 1: Current (Production Ready)
```
RBPF + Omori(10) + GPB1 + Hysteresis Smoothing
Latency: 10μs | Regime Acc: 69% | MAE: 0.016
```

#### Stage 2: "Sweet Spot" (Recommended Next)
```
RBPF + Omori(10) + GPB1 + Fixed-Lag(K=5) + Dual Output
Latency: ~12μs | Expected Regime Acc: 75-80%
```

**Why this works:**
- Fixed-lag smoothing is cheap (circular buffer + re-weighting)
- No extra Kalman updates required
- K=5 provides "regime confirmation" without excessive delay

**Dual Output Strategy:**
```c
typedef struct {
    /* FAST signal (t) - for immediate reactions */
    rbpf_real_t vol_mean;           /* Use for: stop-loss triggers */
    int dominant_regime;            /* Use for: "duck and cover" */
    rbpf_real_t surprise;           /* Use for: anomaly alerts */
    
    /* SMOOTH signal (t-K) - for state-of-world */
    rbpf_real_t vol_mean_smooth;    /* Use for: Kelly sizing */
    int smoothed_regime;            /* Use for: spread adjustment */
    rbpf_real_t regime_confidence;  /* Use for: position limits */
} RBPF_KSC_Output;
```

#### Stage 3: Conditional (Crisis Detection)
```
APF + Omori(10) + GPB1 + Fixed-Lag(K=5)
Latency: ~18-20μs | Best for: Outlier response
```

**APF Value Proposition:**
- Looks ahead at y_t before resampling
- Keeps particles alive that will match incoming data
- Reduces "sample impoverishment" during sudden jumps
- **Worth the 2x latency cost for crisis-sensitive strategies**

**When to use APF:**
- Market-making in volatile assets
- Options trading near expiry
- Any strategy where missing a vol spike is catastrophic

#### Stage 4: SKIP (Diminishing Returns)
```
IMM / GPB2 - NOT RECOMMENDED
Latency: ~100μs | Value: Marginal
```

**Why IMM is a "Hat on a Hat":**

| Component | Purpose | Who Handles It |
|-----------|---------|----------------|
| Particles | Regime uncertainty | ✓ Already covered |
| Kalman | State uncertainty | ✓ Already covered |
| Omori mixture | Observation noise | ✓ Collapsed via GPB1 |
| IMM | Multi-modal state? | ❌ Redundant! |

The Omori 10-component mixture approximates **observation noise** (log(ε²)), not state belief. Log-volatility is locally unimodal within a regime. Multi-modality from regime ambiguity is already captured by having particles distributed across regimes.

**IMM would make sense if:**
- State dynamics were multi-modal (they're not - OU is unimodal)
- You had only 1-2 particles (you have 200)
- Regime switching happened every tick (it doesn't - 90%+ stay probability)

### 11.2 Implementation Priority

| Priority | Enhancement | Effort | Latency | Value |
|----------|-------------|--------|---------|-------|
| **1** | Fixed-Lag(K=5) + Dual Output | 4h | +2μs | High |
| **2** | PMMH offline calibration | 2d | 0 | High |
| **3** | Real data interface | 1d | 0 | Required |
| 4 | Adaptive APF (crisis mode) | 1d | +8μs | Medium |
| ~~5~~ | ~~IMM/GPB2~~ | ~~1w~~ | ~~+90μs~~ | ~~Skip~~ |

### 11.3 Fixed-Lag Smoothing Design

```c
/* Circular buffer for particle ancestry */
#define SMOOTH_LAG 5

typedef struct {
    int parent[SMOOTH_LAG];           /* Ancestry chain */
    rbpf_real_t mu[SMOOTH_LAG];       /* Historical states */
    rbpf_real_t var[SMOOTH_LAG];
    int regime[SMOOTH_LAG];
} ParticleHistory;

/* At time t, output for t-K uses: */
/* 1. Current particle weights (from t) */
/* 2. Historical states (from t-K) */
/* 3. Ancestry to trace which t-K state survived to t */

void compute_smoothed_output(RBPF_KSC *rbpf, int lag, 
                              rbpf_real_t *vol_smooth,
                              int *regime_smooth) {
    /* Trace ancestry back K steps */
    /* Weight historical states by current weights */
    /* This is O(n) - no extra Kalman updates! */
}
```

### 11.4 The HFT Decision Matrix

```
                    Latency Budget
                    Tight (<20μs)    Relaxed (<50μs)
                   ┌─────────────────┬─────────────────┐
    Regime         │                 │                 │
    Accuracy       │  Current        │  + Fixed-Lag    │
    Matters Less   │  (10μs)         │  (12μs)         │
                   ├─────────────────┼─────────────────┤
    Regime         │                 │                 │
    Accuracy       │  + Fixed-Lag    │  + APF          │
    Matters More   │  (12μs)         │  (20μs)         │
                   └─────────────────┴─────────────────┘
```

### 11.5 What NOT to Do

| Anti-Pattern | Why It's Bad |
|--------------|--------------|
| IMM with 200 particles | Redundant - particles already handle multimodality |
| GPB2 (keep all 10 components) | 10x compute for marginal accuracy gain |
| K > 10 fixed-lag | Diminishing returns, signal too stale |
| Online PMCMC | 1000x latency, use offline instead |
| More than 10 Omori components | Omori 10 is already near-optimal |

---

## Appendix A: File Structure

```
3-RBPF/
├── rbpf_ksc.h              # Public API header
├── rbpf_ksc.c              # Implementation
├── RBPF_KSC_DOCUMENTATION.md  # This file
└── CMakeLists.txt          # Build configuration

test/
├── rbpf_ksc_bench.c        # Benchmark and diagnostics
└── CMakeLists.txt
```

## Appendix B: Build Instructions

```bash
# Float precision (default, HFT optimized)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Double precision (validation)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DRBPF_USE_DOUBLE=ON
cmake --build build --config Release

# Run benchmark
./build/bench_rbpf_ksc
```

## Appendix C: References

1. Kim, S., Shephard, N., & Chib, S. (1998). "Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models." *Review of Economic Studies*, 65(3), 361-393.

2. Omori, Y., Chib, S., Shephard, N., & Nakajima, J. (2007). "Stochastic Volatility with Leverage: Fast and Efficient Likelihood Inference." *Journal of Econometrics*, 140(2), 425-449.

3. Liu, J., & West, M. (2001). "Combined Parameter and State Estimation in Simulation-Based Filtering." In *Sequential Monte Carlo Methods in Practice*, Springer.

4. Pitt, M. K., & Shephard, N. (1999). "Filtering via Simulation: Auxiliary Particle Filters." *Journal of the American Statistical Association*, 94(446), 590-599.

5. Doucet, A., de Freitas, N., & Gordon, N. (2001). *Sequential Monte Carlo Methods in Practice*. Springer.

---

*End of Documentation*
