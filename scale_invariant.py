"""
When using absolute prices, when future prices are much higher than in the training set, the predictions stay "anchored" to the low training regime. 
I want to transform the data into a scale-invariant form before feeding it to the LSTM, and then invert the predictions back to actual prices after inference.
"""
import argparse
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def to_scale_invariant(df, fine_df, bars_per_df, N=200, vol_N=200):
    """
    Transform OCHL columns of df into scale-invariant form, using close's MA and vol as baseline.
    Returns a DataFrame with the same shape and columns as input, with OCHL columns replaced by their scale-invariant values:
        (log(col) - logma) / vol
    All other columns are unchanged.

    Trims df and fine_df to the overlapping date range.
    """
    # Trim df and fine_df to overlapping date range
    start = max(df.index.min(), fine_df.index.min())
    end = min(df.index.max(), fine_df.index.max())
    df = df.loc[(df.index >= start) & (df.index <= end)]
    fine_df = fine_df.loc[(fine_df.index >= start) & (fine_df.index <= end)]

    out = df.copy()
    out_fine = fine_df.copy()
    ochl = ['open', 'close', 'high', 'low']

    # Compute baseline and volatility from close
    ma_fine = out_fine['close'].rolling(N*bars_per_df, min_periods=N*bars_per_df).mean()
    logma_fine = np.log(ma_fine)
    log_close_fine = np.log(out_fine['close'].clip(lower=1e-12))
    vol_fine = log_close_fine.diff().rolling(vol_N*bars_per_df, min_periods=vol_N*bars_per_df).std()

    # Transform OCHL columns for fine_df
    for col in ochl:
        log_col_fine = np.log(out_fine[col].clip(lower=1e-12))
        out_fine[col] = (log_col_fine - logma_fine) / vol_fine

    # Drop rows with NaNs from rolling windows
    required = ['close']
    out_fine = out_fine.dropna(subset=required).copy()

    # Map ma_fine to out (coarse) based on bars_per_df
    # For each out index, find the last fine index <= out index, and use its ma_fine
    ma_fine_aligned = []
    fine_idx = out_fine.index
    for idx in out.index:
        # Find all fine indices <= this coarse index
        fine_indices = fine_idx[fine_idx <= idx]
        if len(fine_indices) == 0:
            ma_fine_aligned.append(np.nan)
        else:
            ma_fine_aligned.append(ma_fine.loc[fine_indices[-1]])
    out['ma_from_fine'] = ma_fine_aligned
    out['logma_from_fine'] = np.log(out['ma_from_fine'])
    print(f"out shape: {out.shape}")

    # Use mapped ma_fine for OCHL transformation in out
    log_close_out = np.log(out['close'].clip(lower=1e-12))
    vol_out = log_close_fine.diff().rolling(vol_N*bars_per_df, min_periods=vol_N*bars_per_df).std()
    for col in ochl:
        log_col_out = np.log(out[col].clip(lower=1e-12))
        out[col] = (log_col_out - out['logma_from_fine']) / vol_out

    # Add a horizon value to OCHL columns to ensure positivity
    for col in ochl:
        out_fine[col] = out_fine[col] + 100
        out[col] = out[col] + 100
    # Ensure out has the same columns as df (no extra columns)
    out = out[df.columns]
    return out, out_fine

def main():
    parser = argparse.ArgumentParser(description='Scale-invariant transformation for price data')
    parser.add_argument('--datafile', type=str, default='data/ETH_USDT-1h.feather',
        help='Path to input feather file (e.g., data/ETH_USDT-4h.feather)')
    parser.add_argument('--fine_timeframe', type=str, default='1h',
        help='Timeframe for the fine data (e.g., 1h, 30m)')
    args = parser.parse_args()
    datafile = args.datafile
    fine_datafile = datafile.replace('-4h', f'-{args.fine_timeframe}')  # Fine data file path

    # Load data
    df = pd.read_feather(datafile)
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df['change'] = df['close'].pct_change().fillna(0)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Load fine data
    fine_df = pd.read_feather(fine_datafile)
    fine_df = fine_df[['date', 'open', 'high', 'low', 'close', 'volume']]
    fine_df['change'] = fine_df['close'].pct_change().fillna(0)
    fine_df['date'] = pd.to_datetime(fine_df['date'])
    fine_df.set_index('date', inplace=True)

    # Determine how many fine_df bars per df bar
    # Assume df and fine_df have the same symbol, and are continuous
    # Use the time difference between first two rows to infer frequency
    df_freq = (df.index[1] - df.index[0]).total_seconds()
    fine_freq = (fine_df.index[1] - fine_df.index[0]).total_seconds()
    bars_per_df = int(round(df_freq / fine_freq))

    # ==== Build scale-invariant view ====
    out, out_fine = to_scale_invariant(df, fine_df, bars_per_df, N=200, vol_N=200)

    # ==== Quick looks ====
    # 1) Original price vs. baseline (helps see how dev is defined)
    plt.figure(figsize=(11, 4))
    plt.plot(df.index, df['close'], label='Close')
    plt.title('Price and Baseline')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Deviation (log price minus log baseline) for each OCHL
    for col in ['close']:
        plt.figure(figsize=(11, 3))
        plt.plot(out_fine.index, out_fine[f'{col}'])
        plt.title(f'Fine Deviation = log({col}) - log(MA)')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(11, 3))
        plt.plot(out.index, out[f'{col}'])
        plt.title(f'Coarse Deviation = log({col}) - log(MA)')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
