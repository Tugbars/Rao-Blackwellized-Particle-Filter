# PMMH Integration Guide

## The Problem: Stochastic Volatility Parameters Drift

The 2D particle filter tracks two hidden states:
1. **True price** (observed with noise)
2. **Log-volatility** (completely hidden, inferred from price behavior)

The volatility follows an Ornstein-Uhlenbeck process:

```
log_vol[t+1] = (1-θ) * log_vol[t] + θ * μ_v + σ_v * ε
```

Where:
- `θ` (theta_vol): Mean-reversion speed — how fast vol returns to normal
- `μ_v` (mu_vol): Long-term mean log-volatility — the "normal" vol level
- `σ_v` (sigma_vol): Vol-of-vol — how erratically volatility itself moves

**The problem**: These parameters are estimated from historical data, but markets change. A parameter set calibrated during calm markets will fail during a crisis, and vice versa.

## Why PMMH?

When BOCPD fires, it signals that *something changed* — but not *what* the new parameters should be. Recalibration is necessary, but standard approaches have limitations:

| Approach | Problem |
|----------|---------|
| Offline batch MLE | Too slow, stale by completion |
| Online gradient descent | Gets stuck in local optima, no uncertainty quantification |
| Grid search | Curse of dimensionality (3+ params) |
| **PMMH** | ✓ Handles intractable likelihood, ✓ Bayesian posterior, ✓ Fast enough for online use |

### The Core Problem: Intractable Likelihood

To estimate parameters θ = {drift, μ_v, σ_v}, we need to evaluate how likely the observed data is under different parameter values. In a state-space model, this requires integrating over all possible hidden state trajectories:

```
p(y₁:T | θ) = ∫ p(y₁:T | x₁:T, θ) p(x₁:T | θ) dx₁:T
```

For the stochastic volatility model, x includes both price and log-volatility paths. This integral has no closed form — there are infinitely many paths the hidden volatility could have taken.

### How Particle Filters Estimate Likelihood

A particle filter approximates this integral via Monte Carlo. At each timestep, it:
1. Propagates N particles according to the dynamics
2. Weights them by observation likelihood
3. Normalizes weights

The normalizing constant at each step is an estimate of p(yₜ | y₁:ₜ₋₁, θ). The product over all timesteps gives an estimate of the full likelihood:

```
p̂(y₁:T | θ) = ∏ₜ (1/N) Σᵢ wₜⁱ
```

**Critical property**: This estimate is *unbiased*. The expected value equals the true likelihood:

```
E[p̂(y₁:T | θ)] = p(y₁:T | θ)
```

### Why Noisy Likelihood Still Works

Standard Metropolis-Hastings requires evaluating the exact likelihood ratio:

```
α = p(y|θ') / p(y|θ) × prior(θ') / prior(θ)
```

PMMH substitutes noisy PF estimates:

```
α̂ = p̂(y|θ') / p̂(y|θ) × prior(θ') / prior(θ)
```

The Andrieu-Doucet-Holenstein (2010) insight: if the likelihood estimates are unbiased, the Markov chain *still converges to the correct posterior*. The noise causes some wrong accepts/rejects, but these cancel out in expectation.

Intuitively: the noise in numerator and denominator are independent, so the ratio is unbiased in a log sense. More particles = lower variance = faster convergence, but any N > 0 eventually works.

### What "Handling" Regime Changes Means

After BOCPD fires:

1. **Observation window** contains recent data from the new regime
2. **Prior** is centered on current parameters (which are now wrong)
3. **PMMH runs**: proposes new θ, evaluates likelihood via mini-PF, accepts/rejects
4. **Chain explores** parameter space, concentrating where likelihood is high
5. **Posterior mean** of final samples = new parameter estimates
6. **Apply to main PF**: filter now uses regime-appropriate parameters

The key: PMMH finds parameters that *explain the recent observations well*. If volatility spiked, it finds higher μ_v and σ_v. If drift reversed, it finds new drift.

### Why PMMH Specifically?

| Requirement | How PMMH Satisfies It |
|-------------|----------------------|
| Intractable likelihood | Uses PF estimates (unbiased) |
| Multiple parameters | MH explores joint space naturally |
| Uncertainty quantification | Posterior std available from samples |
| Fast enough for online use | 256 particles × 500 iters ≈ 1-2ms |
| Non-blocking | Runs async in separate thread |
| Adapts to new regime | Likelihood peaks at correct new params |

### Shortcomings and Limitations

PMMH is not perfect. Understanding its limitations is essential for proper use:

| Limitation | Consequence | Mitigation |
|------------|-------------|------------|
| **Variance in likelihood estimates** | Need many MCMC iterations (500+) to average out noise | More particles reduces variance but costs time |
| **Random walk proposals** | Inefficient in high dimensions, slow mixing | Limit to 3 params; adaptive proposal std |
| **Prior dependence** | Bad prior = slow convergence or wrong posterior | Center prior on current params with reasonable std |
| **1-2ms latency** | ~100 ticks of stale parameters after changepoint | Acceptable for regime changes (rare events) |
| **Point estimate only** | Posterior mean discards uncertainty | Could use posterior std for confidence intervals |
| **Assumes stationary regime** | If regime changes during PMMH, estimates contaminated | Cancel and restart if new changepoint detected |
| **Local exploration** | Random walk can't jump across modes | Unlikely issue for unimodal vol posteriors |

### When PMMH Struggles

```
Scenario: Very short observation window (< 100 points)

Problem: 
  - Likelihood surface is flat (many params explain few points)
  - Prior dominates posterior
  - PMMH returns something close to prior, not truth

Solution:
  - Require minimum window size before running PMMH
  - Use tighter prior if data is scarce
```

```
Scenario: Rapid successive changepoints

Problem:
  - PMMH takes 1-2ms
  - If new changepoint arrives before completion, observation 
    window now contains data from TWO regimes
  - Estimated params are some average of both (wrong for either)

Solution:
  - Cancel in-flight PMMH on new changepoint
  - Restart with fresh window
  - This is why the API supports pmmh_job_cancel()
```

### Alternatives Considered

| Method | Why Not Used |
|--------|--------------|
| **Kalman Filter** | Requires linear Gaussian model; SV is nonlinear |
| **Extended Kalman** | Linearization errors compound; SV is highly nonlinear |
| **Unscented Kalman** | Still Gaussian assumption; can't capture vol tails |
| **SMC²** | More principled but 10x slower than PMMH |
| **Variational Bayes** | Faster but biased; underestimates uncertainty |
| **Liu-West Filter** | Joint state-param estimation but degeneracy issues |

PMMH occupies the sweet spot: theoretically sound (exact posterior in limit), practically fast enough (1-2ms), and robust (unbiased estimates).

## System Architecture

PMMH operates as an asynchronous recalibration service within the trading pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Trading System Pipeline                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Market Data                                                           │
│       │                                                                 │
│       ▼                                                                 │
│   ┌───────┐     ┌─────────┐     ┌──────────┐     ┌───────────────┐    │
│   │  SSA  │ ──▶ │  BOCPD  │ ──▶ │   PF2D   │ ──▶ │ Kelly Sizing  │    │
│   └───────┘     └─────────┘     └──────────┘     └───────────────┘    │
│   Trend/Noise   Changepoint      Volatility       Position Size        │
│   Separation    Detection        Estimation                            │
│                      │                ▲                                 │
│                      │                │                                 │
│                      ▼                │                                 │
│                 ┌─────────┐           │                                 │
│                 │  PMMH   │───────────┘                                │
│                 │  (async)│   Returns calibrated {drift, μ_v, σ_v}     │
│                 └─────────┘                                             │
│                   1-2ms                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Input | Output | Role in Pipeline |
|-----------|-------|--------|------------------|
| **SSA** | Raw prices | Trend, noise decomposition | Signal preprocessing |
| **BOCPD** | Price innovations | Changepoint probability | Regime change detection |
| **PMMH** | Recent observations, priors | Posterior {drift, μ_v, σ_v} | Parameter recalibration |
| **PF2D** | Observations, regime params | Filtered price, volatility | State estimation |
| **Kelly** | Volatility estimate, signal | Position size | Risk-adjusted sizing |

### The Recalibration Problem

After a regime change, the particle filter operates with stale parameters:

| Problem | Consequence | Downstream Impact |
|---------|-------------|-------------------|
| **Old μ_v** | Vol estimate reverts to wrong mean | Kelly uses incorrect baseline |
| **Old σ_v** | Vol dynamics too narrow/wide | Filter can't track vol spikes |
| **Old drift** | Price model has wrong trend | Systematic estimation bias |

PMMH addresses this by re-estimating these parameters from recent observations, conditioned on the new regime.

### Concrete Example

```
Pre-crisis:   μ_v = log(0.01)  →  "normal" vol = 1%
              σ_v = 0.05       →  vol is stable
              
Crisis hits:  BOCPD fires
              
PMMH finds:   μ_v = log(0.04)  →  "normal" vol = 4%
              σ_v = 0.15       →  vol is erratic
              
Result:       PF correctly tracks elevated volatility
              Kelly reduces position sizes appropriately
              Risk is controlled
```

Without PMMH, the PF would keep expecting vol to revert to 1%, underestimating risk.

## Why These Specific Parameters?

We recalibrate `{drift, μ_v, σ_v}` but fix `{θ, ρ}`:

| Parameter | Recalibrate? | Reason |
|-----------|--------------|--------|
| drift | ✓ | Regime-specific trend |
| μ_v | ✓ | Mean vol level shifts significantly |
| σ_v | ✓ | Vol-of-vol changes with market stress |
| θ (mean-reversion speed) | ✗ | Slow-moving, requires longer history |
| ρ (correlation) | ✗ | Slow-moving, requires longer history |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Main Thread (15μs tick loop)                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. SSA update                                       │   │
│  │  2. BOCPD update → changepoint?                      │   │
│  │  3. PF2D update (using params_live)                  │   │
│  │  4. Check PMMH completion → apply new params         │   │
│  │  5. Kelly sizing                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼ (on changepoint)                 │
│                    Spawn PMMH Thread                        │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  PMMH Thread (runs ~1-2ms async)                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  for iter = 1 to 500:                                │   │
│  │      θ' = θ + ε                                      │   │
│  │      log_lik = run_mini_PF(observations, θ')         │   │
│  │      accept/reject via MH                            │   │
│  │      adapt proposal std                              │   │
│  │                                                      │   │
│  │  return posterior_mean(last 150 samples)             │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│               atomic_update(params_live)                    │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```c
#include "particle_filter_2d.h"
#include "pf2d_pmmh.h"

// === Initialization ===

// Main particle filter
PF2D *pf = pf2d_create(2048, 4);
pf2d_set_resample_method(pf, PF2D_RESAMPLE_REGULARIZED);

// Observation window (ring buffer)
PMMHObsWindow *obs_window = pmmh_obs_window_create(2000);

// Atomic parameter storage
PMMHParamsAtomic params_atomic;
pmmh_params_atomic_init(&params_atomic, pf);

// PMMH job handle
PMMHJob *pmmh_job = NULL;


// === Per-Tick Loop ===

void on_tick(pf2d_real price) {
    // 1. Add observation to window
    pmmh_obs_window_push(obs_window, price);
    
    // 2. Normal pipeline
    ssa_update(&ssa, price);
    bocpd_update(&bocpd, price);
    PF2DOutput out = pf2d_update(pf, price, &regime_probs);
    
    // 3. Check for PMMH completion
    if (pmmh_job && pmmh_job_is_done(pmmh_job)) {
        PMMHResult result;
        pmmh_job_finish(pmmh_job, &result);
        pmmh_job = NULL;
        
        // Update atomic params
        pmmh_params_atomic_update(&params_atomic, result.target_regime, &result);
        
        // Apply to PF
        pmmh_params_atomic_apply(&params_atomic, pf);
        
        printf("PMMH complete: drift=%.6f mu_vol=%.4f sigma_vol=%.4f (accept=%.2f%%)\n",
               result.posterior_mean.drift,
               result.posterior_mean.mu_vol,
               result.posterior_mean.sigma_vol,
               result.acceptance_rate * 100);
    }
    
    // 4. Trigger PMMH on changepoint
    if (bocpd.changepoint_detected) {
        // Cancel existing job if running
        if (pmmh_job) {
            pmmh_job_cancel(pmmh_job);
            pmmh_job_finish(pmmh_job, NULL);  // Wait and cleanup
            pmmh_job = NULL;
        }
        
        // Configure PMMH
        PMMHConfig cfg;
        pmmh_config_defaults(&cfg);
        cfg.target_regime = out.dominant_regime;
        cfg.window_size = min(1000, pmmh_obs_window_count(obs_window));
        pmmh_config_set_prior_from_pf(&cfg, pf, cfg.target_regime, 0.5);
        
        // Launch async
        pmmh_job = pmmh_start_async(obs_window, &cfg, pf);
    }
    
    // 5. Continue to Kelly...
}


// === Cleanup ===

void shutdown() {
    if (pmmh_job) {
        pmmh_job_cancel(pmmh_job);
        pmmh_job_finish(pmmh_job, NULL);
    }
    pmmh_params_atomic_destroy(&params_atomic);
    pmmh_obs_window_destroy(obs_window);
    pf2d_destroy(pf);
}
```

## API Reference

### Observation Window

```c
// Create ring buffer for observations
PMMHObsWindow* pmmh_obs_window_create(int capacity);

// Add observation
void pmmh_obs_window_push(PMMHObsWindow *win, pf2d_real obs);

// Get current count
int pmmh_obs_window_count(const PMMHObsWindow *win);

// Destroy
void pmmh_obs_window_destroy(PMMHObsWindow *win);
```

### Configuration

```c
// Initialize with defaults
void pmmh_config_defaults(PMMHConfig *cfg);

// Set prior from current PF params
// prior_scale: 0.5 = moderate trust in current params
//              1.0 = loose prior, more exploration
void pmmh_config_set_prior_from_pf(PMMHConfig *cfg, const PF2D *pf, 
                                    int regime, pf2d_real prior_scale);
```

### Async Execution

```c
// Start PMMH in background thread
PMMHJob* pmmh_start_async(const PMMHObsWindow *win, 
                           const PMMHConfig *cfg,
                           const PF2D *pf);

// Check if complete (non-blocking)
int pmmh_job_is_done(const PMMHJob *job);

// Request cancellation (non-blocking)
void pmmh_job_cancel(PMMHJob *job);

// Wait for completion and get results
void pmmh_job_finish(PMMHJob *job, PMMHResult *result);
```

### Parameter Updates

```c
// Thread-safe parameter storage
void pmmh_params_atomic_init(PMMHParamsAtomic *pa, const PF2D *pf);
void pmmh_params_atomic_destroy(PMMHParamsAtomic *pa);

// Called from PMMH thread when complete
void pmmh_params_atomic_update(PMMHParamsAtomic *pa, int regime, 
                                const PMMHResult *result);

// Called from main thread to apply updates
int pmmh_params_atomic_apply(PMMHParamsAtomic *pa, PF2D *pf);
```

## Configuration Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_iterations` | 500 | Total MCMC iterations |
| `n_burnin` | 150 | Discard first N samples |
| `n_particles` | 256 | Particles for likelihood estimation |
| `window_size` | 500 | Observations to use |
| `adaptive_proposal` | 1 | Enable proposal std adaptation |
| `drift_std` | 0.0005 | Initial proposal std |
| `mu_vol_std` | 0.08 | Initial proposal std |
| `sigma_vol_log_std` | 0.02 | Initial proposal std (log-space) |

## Parameters Estimated vs Fixed

| Parameter | Estimated by PMMH | Rationale |
|-----------|-------------------|-----------|
| `drift` | ✅ Yes | Shifts on regime change |
| `mu_vol` | ✅ Yes | Long-term vol level can jump |
| `sigma_vol` | ✅ Yes | Vol-of-vol spikes in turbulence |
| `theta_vol` | ❌ Fixed | Mean-reversion speed is slow-moving |
| `rho` | ❌ Fixed | Price-vol correlation is structural |

## Performance

Expected timing on i9-14900KF with AVX2:

| Configuration | Time |
|---------------|------|
| 256 particles × 500 iters | ~1-2ms |
| 128 particles × 500 iters | ~0.5-1ms |
| 256 particles × 300 iters | ~0.6-1ms |

The PMMH thread runs completely async, so main PF loop remains at 15μs.

## Diagnostics

Check `PMMHResult` after completion:

```c
PMMHResult result;
pmmh_job_finish(job, &result);

// Acceptance rate should be 25-40%
if (result.acceptance_rate < 0.15) {
    printf("Warning: acceptance too low, proposal may be too wide\n");
}
if (result.acceptance_rate > 0.50) {
    printf("Warning: acceptance too high, proposal may be too narrow\n");
}

// Posterior std indicates uncertainty
printf("Posterior: drift=%.6f±%.6f\n", 
       result.posterior_mean.drift,
       result.posterior_std.drift);
```

## Concurrency Notes

1. **Cancellation**: If BOCPD fires while PMMH is running, cancel and restart:
   ```c
   if (bocpd.changepoint_detected && pmmh_job) {
       pmmh_job_cancel(pmmh_job);
       pmmh_job_finish(pmmh_job, NULL);  // Blocks briefly
       pmmh_job = pmmh_start_async(...);  // Fresh window
   }
   ```

2. **Parameter Apply**: Call `pmmh_params_atomic_apply()` from main thread only:
   ```c
   // Main thread only!
   if (pmmh_params_atomic_apply(&params_atomic, pf)) {
       printf("Parameters updated from PMMH\n");
   }
   ```

3. **RNG Isolation**: PMMH uses its own MKL RNG streams, separate from main PF.

## Performance Expectations

| Metric | Value |
|--------|-------|
| PMMH runtime | 1-2ms (256 particles × 500 iters) |
| Main loop impact | Zero (async thread) |
| Parameter latency | ~30-50 ticks after BOCPD |
| Memory overhead | ~2KB observation window + thread stack |

## When to Use PMMH

| Scenario | Use PMMH? | Alternative |
|----------|-----------|-------------|
| BOCPD fires, need new params | ✓ Yes | — |
| Gradual drift detection | Consider | Periodic scheduled recalibration |
| Initial calibration (offline) | ✓ Yes, longer chains | Batch MLE |
| Very high-frequency (>100kHz) | Maybe | Pre-computed regime lookup |

## Summary

PMMH bridges the gap between "we detected a regime change" (BOCPD) and "we have correct model parameters" (PF2D). Without it:

- PF2D uses stale parameters after regime change
- Volatility estimates are biased
- Kelly criterion makes poor sizing decisions
- Risk is mismanaged

With PMMH:

- Regime change → 1-2ms async recalibration
- Updated {drift, μ_v, σ_v} applied atomically
- PF2D tracks new regime accurately
- Kelly adapts position sizes to new reality

## Compilation

```bash
# Compile all PF2D components
gcc -O3 -mavx2 -fopenmp \
    particle_filter_2d.c \
    pf2d_adaptive.c \
    pf2d_pmmh.c \
    main.c \
    -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
    -liomp5 -lpthread -lm \
    -o trading_system
```

## References

- Andrieu, Doucet, Holenstein (2010): "Particle Markov chain Monte Carlo methods"
- Original PMCMC theory proving noisy likelihood estimates still yield correct posterior
