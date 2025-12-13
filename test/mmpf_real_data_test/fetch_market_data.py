#!/usr/bin/env python3
"""
Fetch real market data from yfinance for MMPF integration testing.

Covers known volatility events:
  - COVID crash (Feb-Apr 2020)
  - 2022 Bear market / Fed hiking
  - Recent data (last 7 days, 1-minute)
  - Flash events if available

Output: CSV files in mmpf_test_data subfolder (next to this script)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse


def fetch_daily_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV data."""
    print(f"  Fetching {ticker} daily: {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
    if df.empty:
        print(f"    WARNING: No data returned for {ticker}")
        return pd.DataFrame()
    
    # Handle multi-index columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    
    # yfinance now uses auto_adjust=True by default
    # So 'Close' is already adjusted, no 'Adj Close' column
    if 'Close' in df.columns:
        df = df.rename(columns={'Date': 'timestamp', 'Close': 'close'})
    elif 'Adj Close' in df.columns:
        # Fallback for older yfinance versions
        df = df.rename(columns={'Date': 'timestamp', 'Adj Close': 'close'})
    else:
        print(f"    WARNING: No Close column found. Columns: {df.columns.tolist()}")
        return pd.DataFrame()
    
    return df[['timestamp', 'close']].dropna()


def fetch_intraday_data(ticker: str, period: str = "7d", interval: str = "1m") -> pd.DataFrame:
    """Fetch intraday data (1-minute bars)."""
    print(f"  Fetching {ticker} intraday: {period} @ {interval}...")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        print(f"    WARNING: No intraday data for {ticker}")
        return pd.DataFrame()
    
    # Handle multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    
    # Column name varies: 'Datetime' for intraday, 'Date' for daily
    time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df = df.rename(columns={time_col: 'timestamp', 'Close': 'close'})
    
    if 'close' not in df.columns:
        print(f"    WARNING: No Close column found. Columns: {df.columns.tolist()}")
        return pd.DataFrame()
    
    return df[['timestamp', 'close']].dropna()


def compute_returns(df: pd.DataFrame, log_returns: bool = False) -> pd.DataFrame:
    """Compute simple or log returns from price series."""
    df = df.copy()
    if log_returns:
        df['return'] = np.log(df['close'] / df['close'].shift(1))
    else:
        df['return'] = df['close'].pct_change()
    
    # Drop first row (NaN return)
    df = df.dropna()
    
    return df


def add_event_labels(df: pd.DataFrame, events: dict) -> pd.DataFrame:
    """Add event labels based on date ranges."""
    df = df.copy()
    df['event'] = ''
    
    for label, (start, end) in events.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        mask = (df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)
        df.loc[mask, 'event'] = label
    
    return df


def save_dataset(df: pd.DataFrame, name: str, output_dir: Path):
    """Save dataset to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.csv"
    
    # Format for C consumption
    out_df = df.copy()
    out_df['timestamp'] = out_df['timestamp'].astype(str)
    
    out_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path} ({len(df)} rows)")
    
    # Print statistics
    returns = df['return'].values
    print(f"    Return stats: mean={returns.mean()*100:.4f}%, std={returns.std()*100:.4f}%")
    print(f"    Min={returns.min()*100:.2f}%, Max={returns.max()*100:.2f}%")
    if 'event' in df.columns:
        events = df[df['event'] != '']['event'].value_counts()
        if len(events) > 0:
            print(f"    Events: {dict(events)}")


def fetch_all_datasets(output_dir: Path, ticker: str = "SPY"):
    """Fetch all test datasets."""
    
    print(f"\n{'='*60}")
    print(f"  MMPF Real Data Fetcher")
    print(f"  Ticker: {ticker}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Known volatility events
    events = {
        'COVID_crash': ('2020-02-20', '2020-03-23'),
        'COVID_recovery': ('2020-03-24', '2020-04-30'),
        'Fed_hike_2022': ('2022-01-01', '2022-06-30'),
        'Oct_2022_bottom': ('2022-10-01', '2022-10-31'),
        'SVB_crisis': ('2023-03-08', '2023-03-15'),
        'Aug_2024_selloff': ('2024-08-01', '2024-08-10'),
    }
    
    datasets = []
    
    # 1. COVID Crash Period (daily) - THE classic volatility event
    print("[1/5] COVID Crash Period (Feb-May 2020)")
    df_covid = fetch_daily_data(ticker, "2020-02-01", "2020-05-31")
    if not df_covid.empty:
        df_covid = compute_returns(df_covid)
        df_covid = add_event_labels(df_covid, events)
        save_dataset(df_covid, f"{ticker.lower()}_covid_2020", output_dir)
        datasets.append(('covid_2020', df_covid))
    
    # 2. 2022 Bear Market (daily) - Sustained high volatility
    print("\n[2/5] 2022 Bear Market (Jan-Dec 2022)")
    df_2022 = fetch_daily_data(ticker, "2022-01-01", "2022-12-31")
    if not df_2022.empty:
        df_2022 = compute_returns(df_2022)
        df_2022 = add_event_labels(df_2022, events)
        save_dataset(df_2022, f"{ticker.lower()}_bear_2022", output_dir)
        datasets.append(('bear_2022', df_2022))
    
    # 3. 2023-2024 Recovery + SVB Crisis
    print("\n[3/5] 2023-2024 Period (includes SVB crisis)")
    df_2023_24 = fetch_daily_data(ticker, "2023-01-01", "2024-10-31")
    if not df_2023_24.empty:
        df_2023_24 = compute_returns(df_2023_24)
        df_2023_24 = add_event_labels(df_2023_24, events)
        save_dataset(df_2023_24, f"{ticker.lower()}_2023_2024", output_dir)
        datasets.append(('2023_2024', df_2023_24))
    
    # 4. Recent intraday data (1-minute) - Tests HFT-like conditions
    print("\n[4/5] Recent Intraday (1-minute, last 7 days)")
    df_intraday = fetch_intraday_data(ticker, period="7d", interval="1m")
    if not df_intraday.empty:
        df_intraday = compute_returns(df_intraday)
        save_dataset(df_intraday, f"{ticker.lower()}_intraday_1m", output_dir)
        datasets.append(('intraday_1m', df_intraday))
    
    # 5. Full 5-year history for long-term validation
    print("\n[5/5] Full 5-Year History (daily)")
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
    df_full = fetch_daily_data(ticker, start_date, end_date)
    if not df_full.empty:
        df_full = compute_returns(df_full)
        df_full = add_event_labels(df_full, events)
        save_dataset(df_full, f"{ticker.lower()}_5year", output_dir)
        datasets.append(('5year', df_full))
    
    # Create combined manifest
    manifest_path = output_dir / "manifest.txt"
    with open(manifest_path, 'w') as f:
        f.write(f"# MMPF Test Data Manifest\n")
        f.write(f"# Ticker: {ticker}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
        for name, df in datasets:
            f.write(f"{name}: {len(df)} rows\n")
    
    print(f"\n{'='*60}")
    print(f"  Done! {len(datasets)} datasets saved to {output_dir}")
    print(f"{'='*60}\n")
    
    return datasets


def fetch_multi_asset(output_dir: Path):
    """Fetch data for multiple asset classes."""
    
    assets = {
        'SPY': 'US Equity (S&P 500)',
        'QQQ': 'US Equity (Nasdaq)',
        'IWM': 'US Equity (Small Cap)',
        'TLT': 'US Bonds (20Y Treasury)',
        'GLD': 'Commodities (Gold)',
        'USO': 'Commodities (Oil)',
        'FXE': 'FX (EUR/USD proxy)',
        'BTC-USD': 'Crypto (Bitcoin)',
        'ETH-USD': 'Crypto (Ethereum)',
    }
    
    print("\n" + "="*60)
    print("  Multi-Asset Data Fetch")
    print("="*60)
    
    for ticker, description in assets.items():
        print(f"\n>>> {ticker} - {description}")
        try:
            fetch_all_datasets(output_dir / ticker.replace('-', '_'), ticker)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch market data for MMPF testing")
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--output", default=None, help="Output directory (default: mmpf_test_data in script dir)")
    parser.add_argument("--multi", action="store_true", help="Fetch multiple asset classes")
    
    args = parser.parse_args()
    
    # Default output: mmpf_test_data subfolder next to this script
    if args.output is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "mmpf_test_data"
    else:
        output_dir = Path(args.output)
    
    if args.multi:
        fetch_multi_asset(output_dir)
    else:
        fetch_all_datasets(output_dir, args.ticker)