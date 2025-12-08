# RBPF as Change Detector: Replacing BOCPD

## The Old Stack

```
SSA → BOCPD → PF → Kelly
       ↓       ↓
   (change) (vol)
```

Two separate models, two sets of assumptions, two things to tune.

## The Problem with BOCPD

BOCPD assumes a **constant hazard rate**:

```
P(changepoint at t) = λ    ← same probability every tick
```

Markets don't work this way:
- Volatility clusters (calm periods last, crises cluster)
- Regime duration is state-dependent
- Changes aren't memoryless

BOCPD asks: "Did something change?"  
But it can't tell you **what** changed or **to what**.

## The Insight

RBPF already computes everything needed for change detection. It's not a separate model—it's a **byproduct of filtering**.

### Signal 1: Surprise

```c
marginal_likelihood = Σᵢ wᵢ × p(yₜ | particleᵢ)
surprise = -log(marginal_likelihood)
```

When the observation is unlikely under the current particle distribution, surprise spikes.

```
Normal tick:    p(y|particles) = 0.3    → surprise = 1.2
Anomaly:        p(y|particles) = 0.001  → surprise = 6.9  ← CHANGE
```

**This is exactly what BOCPD computes**, but RBPF does it as part of the Kalman update. Free.

### Signal 2: Vol Ratio

```c
vol_ema_short = 0.1 × vol + 0.9 × vol_ema_short   // Fast
vol_ema_long  = 0.01 × vol + 0.99 × vol_ema_long  // Slow

vol_ratio = vol_ema_short / vol_ema_long
```

```
Stable:     vol_ratio ≈ 1.0
Vol spike:  vol_ratio > 1.5  ← CHANGE
Vol crash:  vol_ratio < 0.5  ← CHANGE
```

### Signal 3: Regime Entropy

```c
regime_entropy = -Σᵣ p(r) × log(p(r))
```

```
Confident:  p = [0.9, 0.05, 0.03, 0.02]  → entropy = 0.4
Uncertain:  p = [0.3, 0.3, 0.2, 0.2]     → entropy = 1.3  ← TRANSITION
```

High entropy = filter is confused = regime boundary.

### Signal 4: Regime Flip

```c
regime_changed = (dominant_regime != prev_regime) && (confidence > 0.7)
```

Structural change with high confidence.

## The Binding

Pipeline interprets these signals:

```c
// Severity
if (surprise >= 5.0 || vol_ratio >= 2.0)
    change_detected = MAJOR;
else if (surprise >= 3.0 || vol_ratio >= 1.5)
    change_detected = MINOR;

// Type
if (vol_spike && regime_shift)  change_type = BOTH;
else if (regime_shift)          change_type = STRUCTURAL;
else if (vol_spike)             change_type = VOL_SPIKE;

// Action
if (change_detected == MAJOR)   position_scale = 0.25;
if (change_detected == MINOR)   position_scale = 0.50;
if (regime_confidence < 0.6)    position_scale *= 0.7;
```

## RBPF vs BOCPD

| Aspect | BOCPD | RBPF |
|--------|-------|------|
| **Assumption** | Constant hazard λ | None (data-driven) |
| **Output** | "Something changed" | "Entering regime 3 at 85% confidence" |
| **Separate model?** | Yes | No (byproduct of filtering) |
| **Tuning** | Hazard rate, prior | Already tuned for vol tracking |
| **Latency** | +30-50μs | +0μs (already computed) |

## The New Stack

```
SSA → RBPF → Kelly
        │
        ├── vol_forecast      → bet sizing
        ├── regime            → parameter selection  
        ├── surprise          → change detection
        ├── vol_ratio         → change detection
        ├── regime_entropy    → uncertainty
        └── position_scale    → risk multiplier
```

One filter. All signals. No BOCPD.

## Why It Works Better

### 1. No Hazard Rate Assumption

BOCPD: "Changes happen at rate λ"  
RBPF: "This observation is unlikely under current state"

RBPF detects changes **when they happen**, not at a predetermined rate.

### 2. Structural Information

BOCPD: "Changepoint detected"  
RBPF: "Transition from regime 1 (normal) to regime 3 (crisis)"

You know **what** changed and **to what**.

### 3. Unified Uncertainty

BOCPD gives changepoint probability.  
RBPF gives:
- Regime probabilities
- Regime confidence
- Regime entropy
- ESS (particle health)

All aspects of uncertainty in one place.

### 4. One Less Thing

```
Old: Tune BOCPD + tune PF + make them talk to each other
New: Tune RBPF (already done for vol tracking)
```

Less knobs. Less assumptions. Less handshakes.

## The Key Insight

> **Change detection is not a separate problem.**
>
> It's a natural output of Bayesian filtering.
> When the model is surprised, something changed.
> RBPF already computes this.

## Summary

```
BOCPD: Separate model with hazard rate assumption
       → "Something changed"
       
RBPF:  Byproduct of filtering, no extra assumptions
       → "Entering regime 3 with 85% confidence"
       → "Surprise = 6.2, vol_ratio = 2.1"
       → "Reduce position to 25%"
```

The marginal likelihood IS a change detector. RBPF computes it anyway. Use it.
