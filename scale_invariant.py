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

def to_scale_invariant(df, N=200, vol_N=200):
    """
    Transform OCHL columns of df into scale-invariant form, using close's MA and vol as baseline.
    Returns a DataFrame with the same shape and columns as input, with OCHL columns replaced by their scale-invariant values:
        (log(col) - logma) / vol
    All other columns are unchanged.
    """
    out = df.copy()
    ochl = ['open', 'close', 'high', 'low']

    # Compute baseline and volatility from close
    ma = out['close'].rolling(N, min_periods=N).mean()
    logma = np.log(ma)
    log_close = np.log(out['close'].clip(lower=1e-12))
    vol = log_close.diff().rolling(vol_N, min_periods=vol_N).std()

    # Transform OCHL columns
    for col in ochl:
        log_col = np.log(out[col].clip(lower=1e-12))
        out[col] = (log_col - logma) / vol

    # Drop rows with NaNs from rolling windows
    required = ['close']
    out = out.dropna(subset=required).copy()
    # Add 1000 to OCHL columns to ensure positivity
    for col in ochl:
        out[col] = out[col] + 1000
    return out

def main():
    parser = argparse.ArgumentParser(description='Scale-invariant transformation for price data')
    parser.add_argument('--datafile', type=str, default='data/ETH_USDT-1h.feather',
        help='Path to input feather file (e.g., data/ETH_USDT-4h.feather)')
    args = parser.parse_args()
    datafile = args.datafile

    # Load data
    df = pd.read_feather(datafile)
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df['change'] = df['close'].pct_change().fillna(0)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # ==== Build scale-invariant view ====
    si = to_scale_invariant(df, N=200, vol_N=200)

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
        plt.plot(si.index, si[f'{col}'])
        plt.title(f'Deviation = log({col}) - log(MA)')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
