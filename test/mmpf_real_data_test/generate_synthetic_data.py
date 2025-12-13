#!/usr/bin/env python3
"""
Generate synthetic market data mimicking real volatility events.

Since yfinance is blocked, we generate realistic test data with:
  - COVID-like crash (Feb-Apr 2020 pattern)
  - Bear market periods
  - Flash crashes
  - Normal calm periods

Uses regime-switching model with realistic parameters.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def generate_sv_returns(n_days: int, regimes: list, seed: int = 42) -> tuple:
    """
    Generate returns using stochastic volatility model with regime switches.
    
    Args:
        n_days: Number of trading days
        regimes: List of (n_days, vol_level, phi, drift) tuples
        seed: Random seed
    
    Returns:
        (returns, log_vols, regime_labels)
    """
    np.random.seed(seed)
    
    returns = []
    log_vols = []
    labels = []
    
    current_log_vol = None
    
    for regime_idx, (days, vol_level, phi, drift) in enumerate(regimes):
        mu_vol = np.log(vol_level)
        sigma_eta = 0.1 if vol_level < 0.02 else 0.2  # More noise in high-vol regimes
        
        if current_log_vol is None:
            current_log_vol = mu_vol
        
        for _ in range(days):
            # SV dynamics: h_t = mu + phi*(h_{t-1} - mu) + sigma_eta * eps
            current_log_vol = mu_vol + phi * (current_log_vol - mu_vol) + sigma_eta * np.random.randn()
            
            # Return: r_t = drift + exp(h_t) * z_t
            vol = np.exp(current_log_vol)
            ret = drift + vol * np.random.randn()
            
            returns.append(ret)
            log_vols.append(current_log_vol)
            labels.append(regime_idx)
    
    return np.array(returns), np.array(log_vols), np.array(labels)


def generate_covid_crash_data(output_dir: Path, seed: int = 2020):
    """Generate data mimicking Feb-May 2020 pattern."""
    
    # Regime parameters: (days, daily_vol, phi, daily_drift)
    regimes = [
        # Pre-crash calm (Feb 1-19): 13 trading days
        (13, 0.008, 0.98, 0.0005),
        
        # Initial selloff (Feb 20 - Mar 9): 13 days, rising vol
        (13, 0.025, 0.90, -0.015),
        
        # Crash peak (Mar 10-23): 10 days, extreme vol
        (10, 0.055, 0.80, -0.02),
        
        # Bounce/recovery (Mar 24 - Apr 15): 17 days
        (17, 0.035, 0.85, 0.01),
        
        # Stabilization (Apr 16 - May 29): 30 days
        (30, 0.018, 0.92, 0.003),
    ]
    
    returns, log_vols, regime_labels = generate_sv_returns(
        sum(r[0] for r in regimes), regimes, seed
    )
    
    # Create timestamps (trading days only)
    start_date = datetime(2020, 2, 3)
    timestamps = []
    current = start_date
    for i in range(len(returns)):
        while current.weekday() >= 5:  # Skip weekends
            current += timedelta(days=1)
        timestamps.append(current)
        current += timedelta(days=1)
    
    # Event labels
    events = []
    for i, label in enumerate(regime_labels):
        if label == 0:
            events.append('')
        elif label == 1:
            events.append('COVID_selloff')
        elif label == 2:
            events.append('COVID_crash')
        elif label == 3:
            events.append('COVID_recovery')
        else:
            events.append('')
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': 100 * np.exp(np.cumsum(returns)),  # Synthetic price
        'return': returns,
        'event': events,
        'true_log_vol': log_vols,
        'true_regime': regime_labels,
    })
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'synthetic_covid_2020.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Generated: {output_path}")
    print(f"  {len(df)} trading days")
    print(f"  Return: mean={returns.mean()*100:.3f}%, std={returns.std()*100:.3f}%")
    print(f"  Min={returns.min()*100:.1f}%, Max={returns.max()*100:.1f}%")
    
    return df


def generate_bear_market_2022(output_dir: Path, seed: int = 2022):
    """Generate data mimicking 2022 bear market."""
    
    # Full year with multiple volatility spikes
    regimes = [
        # Jan: Fed pivot fears
        (20, 0.015, 0.93, -0.003),
        
        # Feb-Mar: Ukraine + rate hikes
        (40, 0.022, 0.90, -0.002),
        
        # Apr-May: Tech selloff
        (40, 0.020, 0.91, -0.004),
        
        # Jun: Capitulation
        (22, 0.028, 0.88, -0.005),
        
        # Jul-Aug: Bear rally
        (44, 0.014, 0.94, 0.004),
        
        # Sep-Oct: New lows
        (44, 0.022, 0.89, -0.003),
        
        # Nov-Dec: Stabilization
        (42, 0.015, 0.93, 0.001),
    ]
    
    returns, log_vols, regime_labels = generate_sv_returns(
        sum(r[0] for r in regimes), regimes, seed
    )
    
    start_date = datetime(2022, 1, 3)
    timestamps = []
    current = start_date
    for i in range(len(returns)):
        while current.weekday() >= 5:
            current += timedelta(days=1)
        timestamps.append(current)
        current += timedelta(days=1)
    
    # Event labels
    events = []
    cumulative = 0
    event_names = ['Fed_pivot', 'Ukraine', 'Tech_selloff', 'Capitulation', 
                   'Bear_rally', 'Oct_low', 'Stabilization']
    for i, (days, _, _, _) in enumerate(regimes):
        for _ in range(days):
            events.append(event_names[i] if i in [0, 2, 3, 5] else '')
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': 100 * np.exp(np.cumsum(returns)),
        'return': returns,
        'event': events,
        'true_log_vol': log_vols,
        'true_regime': regime_labels,
    })
    
    output_path = output_dir / 'synthetic_bear_2022.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Generated: {output_path}")
    print(f"  {len(df)} trading days")
    print(f"  Return: mean={returns.mean()*100:.3f}%, std={returns.std()*100:.3f}%")
    
    return df


def generate_flash_crash(output_dir: Path, seed: int = 2010):
    """Generate data with embedded flash crash event."""
    
    regimes = [
        # Normal period
        (50, 0.010, 0.97, 0.0003),
        
        # Flash crash day (single day extreme vol)
        (1, 0.08, 0.5, -0.05),  # ~8% vol, -5% drift
        
        # Post-crash (elevated vol, quick recovery)
        (5, 0.025, 0.85, 0.005),
        
        # Return to normal
        (44, 0.011, 0.96, 0.0002),
    ]
    
    returns, log_vols, regime_labels = generate_sv_returns(
        sum(r[0] for r in regimes), regimes, seed
    )
    
    start_date = datetime(2010, 4, 1)
    timestamps = []
    current = start_date
    for i in range(len(returns)):
        while current.weekday() >= 5:
            current += timedelta(days=1)
        timestamps.append(current)
        current += timedelta(days=1)
    
    events = [''] * 50 + ['Flash_crash'] + ['Post_crash'] * 5 + [''] * 44
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': 100 * np.exp(np.cumsum(returns)),
        'return': returns,
        'event': events,
        'true_log_vol': log_vols,
        'true_regime': regime_labels,
    })
    
    output_path = output_dir / 'synthetic_flash_crash.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Generated: {output_path}")
    print(f"  Flash crash return: {returns[50]*100:.1f}%")
    
    return df


def generate_intraday_hft(output_dir: Path, n_minutes: int = 5000, seed: int = 12345):
    """Generate 1-minute intraday data for HFT testing."""
    
    # Intraday volatility is ~1/sqrt(390) of daily
    # Daily 1% vol -> 1-min ~0.05% vol
    
    regimes = [
        # Morning: higher vol
        (390, 0.0006, 0.95, 0.0),
        
        # Midday: calm
        (780, 0.0003, 0.98, 0.0),
        
        # News spike
        (30, 0.0015, 0.80, 0.0),
        
        # Afternoon
        (390, 0.0004, 0.96, 0.0),
        
        # Close: elevated
        (200, 0.0005, 0.94, 0.0),
        
        # More calm
        (n_minutes - 1790, 0.0003, 0.98, 0.0),
    ]
    
    # Adjust if total doesn't match
    total = sum(r[0] for r in regimes)
    if total != n_minutes:
        regimes[-1] = (regimes[-1][0] + (n_minutes - total),) + regimes[-1][1:]
    
    returns, log_vols, regime_labels = generate_sv_returns(n_minutes, regimes, seed)
    
    # Timestamps: every minute during market hours (9:30-16:00)
    start_time = datetime(2024, 12, 9, 9, 30)
    timestamps = []
    current = start_time
    for i in range(n_minutes):
        timestamps.append(current)
        current += timedelta(minutes=1)
        # Skip to next day if past 16:00
        if current.hour >= 16:
            current = current.replace(hour=9, minute=30) + timedelta(days=1)
            while current.weekday() >= 5:
                current += timedelta(days=1)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': 100 * np.exp(np.cumsum(returns)),
        'return': returns,
        'event': [''] * n_minutes,
        'true_log_vol': log_vols,
        'true_regime': regime_labels,
    })
    
    output_path = output_dir / 'synthetic_intraday_1m.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Generated: {output_path}")
    print(f"  {len(df)} minutes")
    print(f"  Return: mean={returns.mean()*10000:.3f} bps, std={returns.std()*10000:.3f} bps")
    
    return df


def generate_long_history(output_dir: Path, n_years: int = 5, seed: int = 2019):
    """Generate multi-year history with various regime types."""
    
    np.random.seed(seed)
    
    trading_days_per_year = 252
    n_days = n_years * trading_days_per_year
    
    # Generate regime sequence probabilistically
    regimes = []
    days_remaining = n_days
    
    regime_params = {
        'calm':   (0.009, 0.97, 0.0003),
        'normal': (0.012, 0.95, 0.0002),
        'elevated': (0.018, 0.92, -0.0005),
        'high':   (0.025, 0.88, -0.001),
        'crisis': (0.040, 0.82, -0.003),
    }
    
    regime_names = list(regime_params.keys())
    
    while days_remaining > 0:
        # Random regime duration (20-80 days)
        duration = min(days_remaining, np.random.randint(20, 80))
        
        # Pick regime (weighted toward calm/normal)
        weights = [0.35, 0.35, 0.15, 0.10, 0.05]
        regime = np.random.choice(regime_names, p=weights)
        vol, phi, drift = regime_params[regime]
        
        regimes.append((duration, vol, phi, drift))
        days_remaining -= duration
    
    returns, log_vols, regime_labels = generate_sv_returns(n_days, regimes, seed)
    
    start_date = datetime(2019, 1, 2)
    timestamps = []
    current = start_date
    for i in range(len(returns)):
        while current.weekday() >= 5:
            current += timedelta(days=1)
        timestamps.append(current)
        current += timedelta(days=1)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': 100 * np.exp(np.cumsum(returns)),
        'return': returns,
        'event': [''] * len(returns),
        'true_log_vol': log_vols,
        'true_regime': regime_labels,
    })
    
    output_path = output_dir / 'synthetic_5year.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Generated: {output_path}")
    print(f"  {len(df)} trading days ({n_years} years)")
    print(f"  {len(regimes)} regime periods")
    
    return df


if __name__ == "__main__":
    output_dir = Path('./mmpf_test_data')
    
    print("="*60)
    print("  Synthetic Market Data Generator")
    print("="*60 + "\n")
    
    # Generate all datasets
    generate_covid_crash_data(output_dir)
    print()
    
    generate_bear_market_2022(output_dir)
    print()
    
    generate_flash_crash(output_dir)
    print()
    
    generate_intraday_hft(output_dir)
    print()
    
    generate_long_history(output_dir)
    
    print("\n" + "="*60)
    print("  All datasets generated!")
    print("="*60)
