#!/usr/bin/env python3
"""
Visualize MMPF results from real market data test.

Generates multi-panel plots showing:
  - Price/returns with regime overlay
  - MMPF volatility estimate vs realized vol
  - Regime probabilities over time
  - Outlier detection
  - ESS health
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from pathlib import Path
import argparse
from datetime import datetime


# Regime colors
REGIME_COLORS = {
    0: '#2ecc71',  # Calm - green
    1: '#f39c12',  # Trend - orange
    2: '#e74c3c',  # Crisis - red
}

REGIME_NAMES = {
    0: 'Calm',
    1: 'Trend', 
    2: 'Crisis',
}


def load_results(filepath: str) -> pd.DataFrame:
    """Load MMPF output CSV."""
    df = pd.read_csv(filepath)
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Compute realized vol (rolling 20-period std of returns)
    df['realized_vol'] = df['return'].rolling(20).std()
    
    # Compute cumulative return for price proxy
    df['cum_return'] = (1 + df['return']).cumprod()
    
    return df


def plot_overview(df: pd.DataFrame, title: str = "MMPF Real Data Results"):
    """Create multi-panel overview plot."""
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Color bars for regime background
    def add_regime_background(ax, df):
        for i in range(len(df) - 1):
            regime = df['dominant'].iloc[i]
            ax.axvspan(df.index[i], df.index[i+1], 
                      alpha=0.3, color=REGIME_COLORS.get(regime, 'gray'),
                      linewidth=0)
    
    # Panel 1: Cumulative returns with regime overlay
    ax1 = axes[0]
    add_regime_background(ax1, df)
    ax1.plot(df.index, df['cum_return'], 'k-', linewidth=0.8, label='Price')
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title('Price with Regime Overlay')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add regime legend
    legend_elements = [Patch(facecolor=REGIME_COLORS[r], alpha=0.5, label=REGIME_NAMES[r]) 
                      for r in [0, 1, 2]]
    ax1.legend(handles=legend_elements, loc='upper right', ncol=3)
    
    # Panel 2: MMPF Volatility vs Realized
    ax2 = axes[1]
    ax2.plot(df.index, df['vol'] * 100, 'b-', linewidth=0.8, label='MMPF Vol', alpha=0.8)
    ax2.plot(df.index, df['realized_vol'] * 100, 'r--', linewidth=0.8, 
             label='Realized Vol (20-period)', alpha=0.6)
    ax2.fill_between(df.index, 
                     (df['vol'] - df['vol_std']) * 100,
                     (df['vol'] + df['vol_std']) * 100,
                     alpha=0.2, color='blue', label='±1σ')
    ax2.set_ylabel('Volatility (%)')
    ax2.set_title('MMPF Volatility Estimate vs Realized')
    ax2.legend(loc='upper right', ncol=3)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, None)
    
    # Panel 3: Regime Probabilities
    ax3 = axes[2]
    ax3.fill_between(df.index, 0, df['w_calm'], 
                     color=REGIME_COLORS[0], alpha=0.7, label='Calm')
    ax3.fill_between(df.index, df['w_calm'], df['w_calm'] + df['w_trend'],
                     color=REGIME_COLORS[1], alpha=0.7, label='Trend')
    ax3.fill_between(df.index, df['w_calm'] + df['w_trend'], 1,
                     color=REGIME_COLORS[2], alpha=0.7, label='Crisis')
    ax3.set_ylabel('Probability')
    ax3.set_title('Regime Probabilities')
    ax3.set_ylim(0, 1)
    ax3.legend(loc='upper right', ncol=3)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Outlier Detection
    ax4 = axes[3]
    ax4.fill_between(df.index, 0, df['outlier_frac'], 
                     color='purple', alpha=0.5)
    ax4.axhline(0.5, color='red', linestyle='--', linewidth=1, label='50% threshold')
    ax4.set_ylabel('Outlier Fraction')
    ax4.set_title('OCSN Outlier Detection')
    ax4.set_ylim(0, 1)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: ESS and Latency
    ax5 = axes[4]
    ax5.plot(df.index, df['ess_min'], 'g-', linewidth=0.8, alpha=0.7)
    ax5.axhline(50, color='orange', linestyle='--', linewidth=1, label='ESS=50')
    ax5.axhline(20, color='red', linestyle='--', linewidth=1, label='ESS=20 (collapse)')
    ax5.set_ylabel('Min ESS')
    ax5.set_xlabel('Time')
    ax5.set_title('Filter Health (Effective Sample Size)')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    # Format x-axis
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_event_analysis(df: pd.DataFrame, event_name: str):
    """Plot focused view of a specific event period."""
    
    if 'event' not in df.columns:
        print(f"No event column in data")
        return None
    
    # Filter to event period
    event_df = df[df['event'] == event_name]
    if len(event_df) == 0:
        print(f"No data for event: {event_name}")
        return None
    
    # Extend window by 20% on each side for context
    start_idx = df.index.get_loc(event_df.index[0])
    end_idx = df.index.get_loc(event_df.index[-1])
    window = end_idx - start_idx
    start_idx = max(0, start_idx - window // 5)
    end_idx = min(len(df) - 1, end_idx + window // 5)
    
    window_df = df.iloc[start_idx:end_idx+1]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'MMPF Analysis: {event_name}', fontsize=14, fontweight='bold')
    
    # Highlight event period
    def add_event_highlight(ax):
        ax.axvspan(event_df.index[0], event_df.index[-1], 
                  alpha=0.2, color='yellow', label='Event Period')
    
    # Panel 1: Returns
    ax1 = axes[0]
    add_event_highlight(ax1)
    colors = [REGIME_COLORS.get(r, 'gray') for r in window_df['dominant']]
    ax1.bar(window_df.index, window_df['return'] * 100, color=colors, alpha=0.7, width=0.8)
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Daily Returns (colored by regime)')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Volatility
    ax2 = axes[1]
    add_event_highlight(ax2)
    ax2.plot(window_df.index, window_df['vol'] * 100, 'b-', linewidth=2, label='MMPF Vol')
    ax2.fill_between(window_df.index,
                     (window_df['vol'] - window_df['vol_std']) * 100,
                     (window_df['vol'] + window_df['vol_std']) * 100,
                     alpha=0.3, color='blue')
    ax2.set_ylabel('Volatility (%)')
    ax2.set_title('MMPF Volatility Estimate')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Regime probabilities
    ax3 = axes[2]
    add_event_highlight(ax3)
    ax3.plot(window_df.index, window_df['w_calm'], color=REGIME_COLORS[0], 
             linewidth=2, label='Calm')
    ax3.plot(window_df.index, window_df['w_trend'], color=REGIME_COLORS[1],
             linewidth=2, label='Trend')
    ax3.plot(window_df.index, window_df['w_crisis'], color=REGIME_COLORS[2],
             linewidth=2, label='Crisis')
    ax3.set_ylabel('Probability')
    ax3.set_xlabel('Date')
    ax3.set_title('Regime Probabilities')
    ax3.legend(loc='upper right', ncol=3)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_latency_histogram(df: pd.DataFrame):
    """Plot latency distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    latencies = df['latency_us'].values
    
    ax.hist(latencies, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(latencies), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(latencies):.1f} μs')
    ax.axvline(np.percentile(latencies, 99), color='orange', linestyle='--',
               linewidth=2, label=f'P99: {np.percentile(latencies, 99):.1f} μs')
    
    ax.set_xlabel('Latency (μs)')
    ax.set_ylabel('Count')
    ax.set_title('MMPF Step Latency Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stats text
    stats_text = (f"Min: {np.min(latencies):.1f} μs\n"
                  f"Max: {np.max(latencies):.1f} μs\n"
                  f"Std: {np.std(latencies):.1f} μs\n"
                  f"Throughput: {1e6/np.mean(latencies):.0f}/sec")
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontfamily='monospace')
    
    plt.tight_layout()
    return fig


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("  MMPF Results Summary")
    print("="*60 + "\n")
    
    print(f"  Data period: {df.index[0]} to {df.index[-1]}")
    print(f"  Total ticks: {len(df)}")
    print()
    
    print("  Return Statistics:")
    print(f"    Mean:   {df['return'].mean()*100:7.4f}%")
    print(f"    Std:    {df['return'].std()*100:7.4f}%")
    print(f"    Min:    {df['return'].min()*100:7.2f}%")
    print(f"    Max:    {df['return'].max()*100:7.2f}%")
    print()
    
    print("  MMPF Volatility:")
    print(f"    Mean:   {df['vol'].mean()*100:7.4f}%")
    print(f"    Min:    {df['vol'].min()*100:7.4f}%")
    print(f"    Max:    {df['vol'].max()*100:7.4f}%")
    print()
    
    print("  Regime Distribution:")
    regime_counts = df['dominant'].value_counts().sort_index()
    for regime, count in regime_counts.items():
        pct = 100 * count / len(df)
        print(f"    {REGIME_NAMES.get(regime, f'R{regime}'):8s}: {count:5d} ({pct:5.1f}%)")
    print()
    
    print("  Regime Transitions:", df['dominant'].diff().ne(0).sum() - 1)
    print()
    
    if 'event' in df.columns:
        events = df[df['event'] != '']['event'].value_counts()
        if len(events) > 0:
            print("  Events in data:")
            for event, count in events.items():
                print(f"    {event}: {count} ticks")
            print()
    
    print("  Latency:")
    print(f"    Mean: {df['latency_us'].mean():.1f} μs")
    print(f"    P99:  {df['latency_us'].quantile(0.99):.1f} μs")
    print(f"    Max:  {df['latency_us'].max():.1f} μs")
    print()


def main():
    parser = argparse.ArgumentParser(description="Visualize MMPF real data results")
    parser.add_argument("input", help="MMPF output CSV file")
    parser.add_argument("--output-dir", default="./plots", help="Output directory for plots")
    parser.add_argument("--event", help="Focus on specific event (e.g., 'COVID_crash')")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.input}...")
    df = load_results(args.input)
    
    # Print summary
    print_summary_stats(df)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    base_name = Path(args.input).stem
    
    # Overview plot
    print("Generating overview plot...")
    fig_overview = plot_overview(df, f"MMPF Results: {base_name}")
    fig_overview.savefig(output_dir / f"{base_name}_overview.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/{base_name}_overview.png")
    
    # Latency histogram
    print("Generating latency histogram...")
    fig_latency = plot_latency_histogram(df)
    fig_latency.savefig(output_dir / f"{base_name}_latency.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/{base_name}_latency.png")
    
    # Event analysis (if specified or if events exist in data)
    if args.event:
        print(f"Generating event analysis for '{args.event}'...")
        fig_event = plot_event_analysis(df, args.event)
        if fig_event:
            fig_event.savefig(output_dir / f"{base_name}_{args.event}.png", 
                            dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_dir}/{base_name}_{args.event}.png")
    elif 'event' in df.columns:
        # Auto-generate for all events
        events = df[df['event'] != '']['event'].unique()
        for event in events:
            print(f"Generating event analysis for '{event}'...")
            fig_event = plot_event_analysis(df, event)
            if fig_event:
                safe_name = event.replace(' ', '_').replace('/', '_')
                fig_event.savefig(output_dir / f"{base_name}_{safe_name}.png",
                                dpi=150, bbox_inches='tight')
                print(f"  Saved: {output_dir}/{base_name}_{safe_name}.png")
    
    if not args.no_show:
        plt.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
