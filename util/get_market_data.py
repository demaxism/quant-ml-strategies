#!/usr/bin/env python3
# get_usdjpy.py
import argparse
import pandas as pd
import yfinance as yf

"""
modify: ticker = "NFLX" to change ticker
after download the csv, remove the 2nd line
"""

def fetch_usdjpy(period="max", interval="1d", start=None, end=None):
    """
    Fetch USD/JPY OHLCV from Yahoo Finance (ticker: USDJPY=X).
    Note: Forex volume on Yahoo is often 0 or missing; we fill with 0.
    """
    ticker = "USDJPY=X"
    if start or end:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    else:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)

    if df.empty:
        raise RuntimeError("No data returned. Try a different period/interval or provide start/end dates.")

    df = df.reset_index()

    # Normalize column names
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })

    # Keep only the required columns; fill missing volume with 0 for FX
    if "volume" not in df.columns:
        df["volume"] = 0
    df["volume"] = df["volume"].fillna(0).astype("int64")

    # Ensure ISO8601 string for date
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).astype(str)

    # Reorder to requested order: date, open, close, high, low, volume
    df = df[["date", "open", "close", "high", "low", "volume"]]
    ticker = 'USD_JPY'
    return df, ticker

def fetch_stock(ticker="MSFT", period="max", interval="1d", start=None, end=None):
    # MSFT or AMD
    if start or end:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    else:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)

    if df.empty:
        raise RuntimeError("No data returned. Try a different period/interval or specify dates.")

    df = df.reset_index()
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).astype(str)

    # Keep only date, open, close, high, low, volume (your order)
    df = df[["date", "open", "close", "high", "low", "volume"]]
    return df

def main():
    p = argparse.ArgumentParser(description="Download USD/JPY historical data (OHLCV) from Yahoo Finance.")
    p.add_argument("--period", default="max", help="Data period if no start/end (e.g. 1mo, 3mo, 1y, 5y, max).")
    p.add_argument("--interval", default="1d", help="Bar interval (e.g. 1d, 1h, 30m).")
    p.add_argument("--start", help="Start date (YYYY-MM-DD).")
    p.add_argument("--end", help="End date (YYYY-MM-DD).")
    p.add_argument("--ticker", help="Stock ticker: MSFT, META")
    p.add_argument("--out", default="USDJPY_ohlcv.csv", help="Output CSV path.")
    args = p.parse_args()

    # df = fetch_usdjpy(period=args.period, interval=args.interval, start=args.start, end=args.end)
    df = fetch_stock(ticker=args.ticker, period=args.period, interval="1d")
    df.to_csv(f"{args.ticker}.csv", index=False)
    print(f"Saved {len(df):,} rows to {args.out}")
    print(df.head())

if __name__ == "__main__":
    main()