# Particle Rejuvenation: APF and PMMH

## The Core Problem

Standard particle filters suffer from **degeneracy**: resampling kills particles that have low weight *now* but would be valuable *later*.

```
t=99:  Crisis brewing, but current obs still calm
       Particles: [calm, calm, calm, calm, crisis]
       Weights:   [0.25, 0.25, 0.25, 0.24, 0.01]  ← crisis particle dying
       
       Standard resample → kills crisis particle
       
t=100: Crisis hits!
       Filter has no crisis particles → slow adaptation
```

## The Solution: Look Ahead

Both APF and PMMH solve this by **using future information to guide current selection**.

### APF: 1-Step Lookahead

```
At time t, before resampling:

1. Peek at y_{t+1} (available from SSA window)
2. Ask each particle: "how well do you predict this?"
3. Boost particles that predict well
4. Then resample

     Current obs          Future obs
         ↓                    ↓
    ┌─────────┐          ┌─────────┐
    │ Update  │          │ Score   │
    │ weights │    +     │ predict │  =  Combined weights
    └─────────┘          └─────────┘
    
    Particle with low current weight but good prediction → survives
```

### PMMH: Full Trajectory

```
For parameter θ:

1. Run entire particle filter with θ
2. Compute likelihood of full trajectory
3. Accept/reject θ based on trajectory fit

    ┌─────────────────────────────────────┐
    │  Run PF:  y₁ → y₂ → ... → yₜ       │
    │  Compute: p(y₁:ₜ | θ)               │
    └─────────────────────────────────────┘
                      ↓
              Accept θ if trajectory likely
```

## Comparison

| Aspect | APF | PMMH |
|--------|-----|------|
| **Lookahead** | 1 step | Full trajectory |
| **What's rejuvenated** | Particle states | Parameters |
| **When** | Every timestep | Batch/offline |
| **Cost** | +8μs per step | Minutes/hours |
| **Use case** | Real-time filtering | Parameter estimation |

## Why SSA + APF Works

APF's lookahead is only as good as the data:

```
Raw tick t+1:   Bid-ask bounce, +5% fake spike
APF thinks:     "Vol exploding!" → boosts wrong particles → WORSE

SSA-cleaned t+1: True signal, +0.3% real move  
APF thinks:     "Moderate move" → correct boosting → BETTER
```

**SSA provides reliable lookahead. APF exploits it.**

## The Unified Insight

> Don't evaluate particles only on the past.
> Use knowledge of the future to pick survivors.

Standard PF is myopic—it only sees `t`. 

APF sees `t+1`. PMMH sees `1:T`.

Both achieve the same goal: **keep particles that will matter**.

## In Practice

```c
// SSA gives you clean [t, t+1]
// APF uses both

rbpf_pipeline_step_apf(pipe, 
    ssa_clean[t],      // Current: update filter
    ssa_clean[t+1],    // Lookahead: guide resampling
    &signal);

// Particles that predict t+1 well survive
// Even if their current weight is low
```

## Summary

```
Problem:  Resampling kills future-useful particles
          
APF:      Peek at t+1 → boost good predictors → resample
PMMH:     Run full trajectory → keep good parameters

Both:     Use future to guide current selection
          = Particle rejuvenation
```
