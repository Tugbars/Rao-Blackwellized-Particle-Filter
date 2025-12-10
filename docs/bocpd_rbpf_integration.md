# BOCPD + RBPF Integration

## System Architecture

```
Market data (returns)
        │
        ▼
   ┌─────────┐
   │  BOCPD  │ ──► Change signal (posterior collapse detected)
   └─────────┘
        │
        ▼
   ┌─────────┐
   │  RBPF   │ ──► σ_t estimate (continuous log-volatility)
   └─────────┘
        │
        ▼
   ┌─────────┐
   │  Kelly  │ ──► Position size f* = edge / σ_t²
   └─────────┘
```

**RBPF** tracks volatility. **BOCPD** tells RBPF when to adapt faster.

## The Hazard Rate Problem

Standard BOCPD has a fixed hazard rate H - the prior probability of a changepoint at any tick.

| H Value | Behavior | Problem |
|---------|----------|---------|
| 1/50 (aggressive) | Fast detection | False positives on outliers |
| 1/500 (conservative) | Robust to noise | Slow to detect real changes |

The fundamental issue: H assumes changepoints are Poisson-distributed, which markets are not. Volatility clusters, regimes persist for variable durations, and transition rates themselves vary by regime.

## Options Considered

### Option 1: Regime-Dependent Hazard
```c
float H = (current_regime >= 2) ? 1.0f/50 : 1.0f/200;
```
- Higher H in crisis (regimes change faster)
- Lower H in calm (more stability)
- **Problem**: Still a fixed prior, just with more knobs

### Option 2: Multiple Hazard Ensemble
Run parallel BOCPD instances with different H values:
```c
BOCPD fast_detector;   // H = 1/50
BOCPD slow_detector;   // H = 1/500
// Flag change when both agree, or weight by confidence
```
- **Problem**: 2x compute, still using fixed H values

### Option 3: Adaptive Hazard (Online Learning)
Estimate H from recent changepoint frequency using PMMH.
- **Problem**: Theoretically elegant, practically useless. By the time you've estimated H accurately, the market has moved on. H probably isn't even stationary.

### Option 4: Hazard-Free Detection ✓

**Don't rely on H at all.** Instead, monitor how the run-length posterior evolves.

The key insight:
- **Normal tick**: Posterior shifts by +1 (run continues)
- **Outlier**: Posterior gets noisier but doesn't collapse  
- **Real change**: Posterior mass suddenly piles up at r=0,1,2... 

We detect *evidence accumulation rate*, not prior-weighted probability.

## Option 4: Implementation

### Core Idea

Track the mass in the "short run-length" region and detect rapid shifts:

```c
#define SHORT_RUN_WINDOW 10
#define COLLAPSE_THRESHOLD 0.3f

float prev_short_mass = 0.0f;

bool bocpd_detect_change(float* run_length_posterior, int max_r) {
    // Sum mass in short run-length region
    float short_mass = 0.0f;
    for (int r = 0; r < SHORT_RUN_WINDOW && r < max_r; r++) {
        short_mass += run_length_posterior[r];
    }
    
    // Detect rapid posterior collapse
    float delta = short_mass - prev_short_mass;
    prev_short_mass = short_mass;
    
    // Large positive delta = posterior collapsing = likely changepoint
    return (delta > COLLAPSE_THRESHOLD);
}
```

### Why This Works

| Scenario | P(r < 10) before | P(r < 10) after | Delta | Detection |
|----------|------------------|-----------------|-------|-----------|
| Normal tick | 0.05 | 0.05 | 0.00 | No |
| Single outlier | 0.05 | 0.15 | 0.10 | No |
| Outlier burst (5 ticks) | 0.05 | 0.25 | 0.20 | No |
| Real regime change | 0.05 | 0.70 | 0.65 | **Yes** |

The threshold (0.3) catches real changes while filtering outlier bursts.

### Tunable Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `SHORT_RUN_WINDOW` | 10 | Larger = more lag, more robustness |
| `COLLAPSE_THRESHOLD` | 0.3 | Lower = more sensitive, more false positives |

These are much more interpretable than hazard rate H.

## RBPF Integration Points

When BOCPD detects a change, RBPF can respond in several ways:

### 1. Boost Transition Probabilities (Primary)
```c
if (bocpd_change_detected) {
    rbpf_ext_set_change_detected(ext, true);
    // Internally: transition matrix becomes less sticky
    // Normal: 97% stay in regime
    // Boosted: 70% stay in regime
}
```

### 2. Increase Process Noise (Secondary)
```c
if (bocpd_change_detected) {
    rbpf_ext_set_sigma_vol_scale(ext, 2.0f);
    // Kalman filters adapt faster for next N ticks
}
```

### 3. Redistribute Particles (Optional)
```c
if (bocpd_change_detected) {
    // Move some particles to other regimes
    // Prevents being stuck in wrong regime
    rbpf_redistribute_particles(ext);
}
```

### Cooldown

After a change is detected, suppress further detections for N ticks:
```c
#define CHANGE_COOLDOWN 20

static int cooldown_remaining = 0;

bool process_tick(float y) {
    if (cooldown_remaining > 0) {
        cooldown_remaining--;
        return false;
    }
    
    bool change = bocpd_detect_change(posterior, max_r);
    if (change) {
        cooldown_remaining = CHANGE_COOLDOWN;
    }
    return change;
}
```

## Expected Improvements

With BOCPD integration, expect:

| Metric | Without BOCPD | With BOCPD | Why |
|--------|---------------|------------|-----|
| Phase 6 movement | ~0.5 (bad) | ~0.2 | Posterior doesn't collapse on outliers |
| Phase 7 adaptation | ~100 ticks | ~20 ticks | Change signal boosts transitions |
| Regime accuracy | 43% | 55-65% | Faster regime switching |
| Vol tracking corr | 0.82 | 0.85-0.90 | Less lag during transitions |

## Summary

BOCPD's role is simple: **smoke detector**, not firefighter.

- BOCPD says "something changed" (binary signal)
- RBPF figures out what changed and tracks the new state
- Option 4 (posterior collapse detection) avoids the hazard rate problem entirely

The threshold-based approach is:
- Robust to outliers (no collapse = no signal)
- Fast on real changes (collapse is immediate)
- Interpretable (watching evidence, not tuning priors)
