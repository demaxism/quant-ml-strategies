#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Dual-Model Backtester (Trend Following + Mean Reversion)

Usage examples:
  python3 crypto_dual_backtest.py --data BTC_USDT-1h.csv --tz UTC --strategy turtle --fee 0.0006
  python3 crypto_dual_backtest.py --data BTC_USDT-1h.feather --strategy bb_revert --initial-cash 10000
  python3 crypto_dual_backtest.py --data BTC_USDT-1h.csv --plot
  python crypto_dual_backtest.py --data data/ETH_USDT-4h.feather --bt-from 2022-01-01 --strategy turtle --fee 0.0007 --plot

Data format (CSV or Feather): must include columns:
  date, open, high, low, close, volume
Date will be parsed as timezone-aware if possible.
"""

import argparse
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

# -------------------------------
# Utilities
# -------------------------------

def load_ohlcv(path: str, tz: str = "UTC") -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in [".feather", ".ft"]:
        df = pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    required = ["date", "open", "high", "low", "close", "volume"]
    for r in required:
        if r not in [c.lower() for c in df.columns]:
            raise ValueError(f"Missing column '{r}' in data file")
    # Rename to lower
    df = df.rename(columns={c: c.lower() for c in df.columns})
    # Parse date
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Some dates could not be parsed. Please check the file.")
    if tz:
        try:
            df["date"] = df["date"].dt.tz_convert(tz)
        except Exception:
            # If not timezone-aware, localize first
            df["date"] = df["date"].dt.tz_localize("UTC").dt.tz_convert(tz)
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date","open","high","low","close","volume"]]

def compute_buyhold(df: pd.DataFrame, initial_cash: float, fee: float) -> float:
    """Return final equity if buying once at first close (with fee) and selling at last close (with fee)."""
    if len(df) < 2:
        return initial_cash
    buy_price = df["close"].iloc[0] * (1 + fee)
    units = initial_cash / buy_price
    sell_price = df["close"].iloc[-1] * (1 - fee)
    return units * sell_price

def annualized_return(equity_series: pd.Series) -> float:
    """CAGR based on first and last equity and elapsed years."""
    if len(equity_series) < 2:
        return 0.0
    start_val = float(equity_series.iloc[0])
    end_val = float(equity_series.iloc[-1])
    if start_val <= 0 or end_val <= 0:
        return 0.0
    # Approx years by time delta between first and last index row
    # The index is the integer pos; we need dates. We'll pass dates to function instead.
    return 0.0

def max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """Return (max_dd, max_dd_duration_bars)."""
    peaks = equity.cummax()
    drawdowns = (equity - peaks) / peaks
    max_dd = drawdowns.min() if len(drawdowns) else 0.0
    # Duration: bars since last peak
    duration = 0
    max_duration = 0
    for i in range(len(equity)):
        if equity.iloc[i] == peaks.iloc[i]:
            duration = 0
        else:
            duration += 1
            if duration > max_duration:
                max_duration = duration
    return float(max_dd), float(max_duration)

def sharpe_ratio(returns: pd.Series, periods_per_year: int) -> float:
    if returns.std(ddof=0) == 0 or len(returns) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std(ddof=0))

# -------------------------------
# Strategy signal generators
# -------------------------------

def turtle_signals(df: pd.DataFrame, entry_lookback: int = 55, exit_lookback: int = 20) -> pd.DataFrame:
    """
    Classic Turtle Trend Following:
     - Entry long when close breaks N-day high (entry_lookback)
     - Exit when close breaks M-day low (exit_lookback)
    Signals are aligned so that trade is executed on next bar's open.
    """
    high_n = df["high"].rolling(entry_lookback, min_periods=entry_lookback).max()
    low_m  = df["low"].rolling(exit_lookback,  min_periods=exit_lookback).min()

    entry = (df["close"] >= high_n.shift(1)).astype(int)   # 1 when breakout
    exit_  = (df["close"] <= low_m.shift(1)).astype(int)    # 1 when breakdown

    out = df.copy()
    out["entry_long"] = entry
    out["exit_long"]  = exit_
    return out

def bb_reversion_signals(df: pd.DataFrame, length: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands Mean-Reversion (long-only):
     - Buy when close < lower band
     - Exit when close >= middle band
    """
    ma = df["close"].rolling(length, min_periods=length).mean()
    std = df["close"].rolling(length, min_periods=length).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std

    entry = (df["close"] < lower.shift(1)).astype(int)
    exit_ = (df["close"] >= ma.shift(1)).astype(int)

    out = df.copy()
    out["bb_ma"] = ma
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["entry_long"] = entry
    out["exit_long"]  = exit_
    return out

# -------------------------------
# Backtester (long/flat)
# -------------------------------

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp | None
    entry_price: float
    exit_price: float | None
    size: float  # units
    pnl: float | None

def backtest_long_only(df: pd.DataFrame,
                       fee: float = 0.0006,
                       slippage: float = 0.0,
                       initial_cash: float = 10000.0,
                       strategy_name: str = "turtle",
                       periods_per_year: int = 365*24,  # for 1h use 365*24, for 1d use 252
                       risk_per_trade: float = 1.0,
                       tp_ladder: bool = False,
                       tp_start: float = 0.20,
                       tp_end: float = 0.60,
                       tp_steps: int = 10,
                       tp_sell_perc: float = 0.10) -> dict:
    """
    Simple bar-by-bar simulator:
      - Execute entries at next bar's open * (1 + fee + slippage)
      - Execute exits at next bar's open * (1 - fee - slippage)
      - Long/flat, single position at a time. Size = all available cash (risk_per_trade fraction).
    """
    df = df.copy().reset_index(drop=True)
    dates = df["date"]
    closes = df["close"]
    opens = df["open"]

    equity = []
    position_size = 0.0  # units
    in_position = False
    cash = initial_cash
    entry_price = None
    entry_time = None
    trades: list[Trade] = []
    exposure_bars = 0

    # Pre-calc returns for Sharpe (equity pct change per bar)
    equity_series = []
    # TP ladder state
    original_size = 0.0
    tp_levels: list[float] = []
    tp_next_idx = 0
    tp_fills = 0
    scaled_out_units = 0.0

    for i in range(1, len(df)):  # start from 1 to have "next open" at i
        # record equity at bar open (mark-to-market)
        mtm = cash + (position_size * closes.iloc[i-1] if in_position else 0.0)
        equity_series.append(mtm)

        # ladder take-profit before strategy exits
        if in_position and tp_ladder and original_size > 0 and entry_price is not None:
            high_px = df["high"].iloc[i]
            high_ret = (high_px / entry_price) - 1.0 if entry_price > 0 else -1.0
            # Process from the next untriggered level; same bar may trigger multiple levels
            while tp_next_idx < len(tp_levels) and high_ret >= tp_levels[tp_next_idx] and position_size > 0:
                level_pct = tp_levels[tp_next_idx]
                px = entry_price * (1.0 + level_pct)
                sell_units = min(position_size, original_size * tp_sell_perc)
                if sell_units > 0:
                    cash += sell_units * px * (1.0 - fee - slippage)
                    position_size -= sell_units
                    tp_fills += 1
                    scaled_out_units += sell_units
                tp_next_idx += 1

        # exit signal evaluated on bar i-1, executed on bar i open
        if in_position and df["exit_long"].iloc[i-1] == 1:
            px = opens.iloc[i] * (1 - fee - slippage)
            cash = cash + position_size * px
            trades[-1].exit_time = dates.iloc[i]
            trades[-1].exit_price = float(px)
            trades[-1].pnl = float(cash - initial_cash if len(trades)==1 else
                                   cash - (initial_cash + sum(t.pnl for t in trades[:-1] if t.pnl is not None)))
            in_position = False
            position_size = 0.0
            # reset TP ladder state
            original_size = 0.0
            tp_levels = []
            tp_next_idx = 0

        # entry signal evaluated on bar i-1, executed on bar i open
        if (not in_position) and df["entry_long"].iloc[i-1] == 1:
            px = opens.iloc[i] * (1 + fee + slippage)
            budget = cash * risk_per_trade  # fraction of available cash
            size = 0.0 if px <= 0 else budget / px
            if size > 0:
                position_size = size
                cash -= size * px
                in_position = True
                entry_price = float(px)
                entry_time = dates.iloc[i]
                # initialize TP ladder for this position
                original_size = float(size)
                if tp_ladder and tp_steps > 0 and tp_end > tp_start:
                    # Define 10 lines as in spec: step = (end - start) / steps, lines at start+step, ..., end
                    step = (tp_end - tp_start) / tp_steps
                    tp_levels = [tp_start + step * k for k in range(1, tp_steps + 1)]
                else:
                    tp_levels = []
                tp_next_idx = 0
                trades.append(Trade(entry_time=entry_time, exit_time=None,
                                    entry_price=entry_price, exit_price=None,
                                    size=float(size), pnl=None))

        if in_position:
            exposure_bars += 1

    # Close any open position at last close
    if in_position:
        px = closes.iloc[-1] * (1 - fee - slippage)
        cash = cash + position_size * px
        trades[-1].exit_time = dates.iloc[-1]
        trades[-1].exit_price = float(px)
        trades[-1].pnl = float(cash - initial_cash if len(trades)==1 else
                               cash - (initial_cash + sum(t.pnl for t in trades[:-1] if t.pnl is not None)))
        in_position = False
        position_size = 0.0
        # reset TP ladder state
        original_size = 0.0
        tp_levels = []
        tp_next_idx = 0

    equity_series.append(cash)
    equity_series = pd.Series(equity_series, index=df["date"].iloc[:len(equity_series)])

    # Metrics
    final_equity = cash
    buyhold_equity = compute_buyhold(df, initial_cash, fee)
    strategy_ret = (final_equity / initial_cash) - 1.0
    buyhold_ret  = (buyhold_equity / initial_cash) - 1.0
    excess_ret   = strategy_ret - buyhold_ret
    trades_closed = [t for t in trades if t.exit_time is not None]
    total_trades = len(trades_closed)
    wins = [t for t in trades_closed if t.exit_price is not None and t.exit_price > t.entry_price]
    win_rate = (len(wins) / total_trades) if total_trades > 0 else 0.0

    # Bar returns for Sharpe
    eq = equity_series.astype(float)
    rets = eq.pct_change().fillna(0.0)
    # Estimate periods_per_year from data frequency
    if len(eq) >= 2:
        dt_seconds = (eq.index[1] - eq.index[0]).total_seconds()
        if dt_seconds > 0:
            # seconds per year ~ 365*24*3600
            periods_per_year = int(round((365*24*3600) / dt_seconds))
    sr = sharpe_ratio(rets, periods_per_year=periods_per_year)
    mdd, mdd_bars = max_drawdown(eq)

    # CAGR using timestamps
    if len(eq) >= 2:
        elapsed_years = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr = (final_equity / float(eq.iloc[0])) ** (1/elapsed_years) - 1 if elapsed_years > 0 else 0.0
    else:
        cagr = 0.0

    exposure_pct = exposure_bars / max(1, len(df))

    out = {
        "strategy": strategy_name,
        "initial_cash": initial_cash,
        "final_equity": final_equity,
        "strategy_return": strategy_ret,
        "buyhold_equity": buyhold_equity,
        "buyhold_return": buyhold_ret,
        "excess_return": excess_ret,
        "excess_return_per_trade": (excess_ret / total_trades) if total_trades > 0 else 0.0,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "sharpe": sr,
        "max_drawdown": mdd,
        "max_drawdown_bars": mdd_bars,
        "cagr": cagr,
        "exposure_pct": exposure_pct,
        "tp_fills": tp_fills,
        "scaled_out_units": scaled_out_units,
        "equity_curve": eq,
        "trades": trades_closed,
    }
    return out

def plot_results(df: pd.DataFrame, result: dict, data_name: str = "", show: bool = True, save_path: str | None = None):
    eq = result["equity_curve"]
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(eq.index, eq.values, label="Strategy Equity", color="tab:blue")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Equity", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["close"], label="Price", color="tab:orange", alpha=0.3)
    ax2.set_ylabel("Price", color="tab:orange")
    ax2.tick_params(axis='y', labelcolor="tab:orange")

    title = f"Equity vs Price - {result['strategy']} - {data_name}" if data_name else f"Equity vs Price - {result['strategy']}"
    plt.title(title)
    fig.tight_layout()
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def print_summary(res: dict):
    print("\n===== Summary =====")
    print(f"Strategy            : {res['strategy']}")
    print(f"Initial Cash        : {res['initial_cash']:.2f}")
    print(f"Final Equity        : {res['final_equity']:.2f}")
    print(f"Strategy Return     : {res['strategy_return']*100:.2f}%")
    print(f"Buy & Hold Equity   : {res['buyhold_equity']:.2f}")
    print(f"Buy & Hold Return   : {res['buyhold_return']*100:.2f}%")
    print(f"Excess Return       : {res['excess_return']*100:.2f}%")
    print(f"Excess Ret / Trade  : {res['excess_return_per_trade']*100:.2f}%")
    print(f"Total Trades        : {res['total_trades']}")
    print(f"Win Rate            : {res['win_rate']*100:.2f}%")
    print(f"Sharpe (approx)     : {res['sharpe']:.2f}")
    print(f"Max Drawdown        : {res['max_drawdown']*100:.2f}%")
    print(f"CAGR                : {res['cagr']*100:.2f}%")
    print(f"Exposure (bars)     : {res['exposure_pct']*100:.2f}%")
    if 'tp_fills' in res:
        print(f"TP Fills (ladder)   : {res['tp_fills']}")
    if 'scaled_out_units' in res:
        print(f"Scaled-out Units    : {res['scaled_out_units']:.6f}")

# -------------------------------
# CSV trade logging
# -------------------------------

def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append({
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "size": t.size,
            "pnl": t.pnl,
        })
    return pd.DataFrame(rows)

def save_trades_csv(trades: list[Trade], filepath: str, extra_cols: dict | None = None):
    df = trades_to_dataframe(trades)
    if extra_cols:
        for k, v in extra_cols.items():
            df[k] = v
    df.to_csv(filepath, index=False)

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV or Feather file with OHLCV")
    ap.add_argument("--strategy", choices=["turtle","bb_revert"], default="turtle", help="Strategy to run")
    ap.add_argument("--fee", type=float, default=0.0006, help="Per trade proportional fee (e.g., 0.0006)")
    ap.add_argument("--slippage", type=float, default=0.0, help="Per trade slippage proportion")
    ap.add_argument("--initial-cash", type=float, default=10000.0, help="Starting cash")
    ap.add_argument("--tz", default="UTC", help="Timezone for timestamps (e.g., Asia/Tokyo)")
    ap.add_argument("--plot", action="store_true", help="Show equity chart")
    ap.add_argument("--save-chart", default=None, help="Path to save equity chart (PNG)")
    ap.add_argument("--entry-lookback", type=int, default=55, help="Turtle entry lookback (bars)")
    ap.add_argument("--exit-lookback", type=int, default=20, help="Turtle exit lookback (bars)")
    ap.add_argument("--bb-length", type=int, default=20, help="BB window length")
    ap.add_argument("--bb-std", type=float, default=2.0, help="BB standard deviations")
    ap.add_argument("--risk-per-trade", type=float, default=1.0, help="Fraction of cash to allocate per entry [0-1]")
    ap.add_argument("--bt-from", type=str, default=None, help="Backtest start date (inclusive, e.g. 2021-01-01)")
    ap.add_argument("--bt-to", type=str, default=None, help="Backtest end date (inclusive, e.g. 2022-01-01)")
    # TP ladder options
    ap.add_argument("--tp-ladder", action="store_true", help="Enable tiered take-profit (scale-out). When enabled, partial sells occur before strategy exits.")
    ap.add_argument("--tp-start", type=float, default=0.20, help="TP start, as return from entry (e.g., 0.20 for +20%)")
    ap.add_argument("--tp-end", type=float, default=0.60, help="TP end, as return from entry (e.g., 0.60 for +60%)")
    ap.add_argument("--tp-steps", type=int, default=10, help="Number of TP lines (e.g., 10). Lines at start+step,...,end with step=(end-start)/steps")
    ap.add_argument("--tp-sell-perc", type=float, default=0.10, help="Sell percent of original size per line (e.g., 0.10 means 10% of original units per line)")
    args = ap.parse_args()

    df = load_ohlcv(args.data, tz=args.tz)

    # Filter by backtest date range if specified
    if args.bt_from is not None:
        bt_from = pd.to_datetime(args.bt_from, utc=True, errors="coerce")
        if bt_from is not pd.NaT:
            df = df[df["date"] >= bt_from]
    if args.bt_to is not None:
        bt_to = pd.to_datetime(args.bt_to, utc=True, errors="coerce")
        if bt_to is not pd.NaT:
            df = df[df["date"] <= bt_to]

    if args.strategy == "turtle":
        sig = turtle_signals(df, entry_lookback=args.entry_lookback, exit_lookback=args.exit_lookback)
        name = f"turtle_{args.entry_lookback}/{args.exit_lookback}"
    else:
        sig = bb_reversion_signals(df, length=args.bb_length, num_std=args.bb_std)
        name = f"bb_revert_{args.bb_length}x{args.bb_std}"

    res = backtest_long_only(sig,
                             fee=args.fee,
                             slippage=args.slippage,
                             initial_cash=args.initial_cash,
                             strategy_name=name,
                             risk_per_trade=args.risk_per_trade,
                             tp_ladder=args.tp_ladder,
                             tp_start=args.tp_start,
                             tp_end=args.tp_end,
                             tp_steps=args.tp_steps,
                             tp_sell_perc=args.tp_sell_perc)

    # Save trades to CSV under data/log
    data_base = os.path.splitext(os.path.basename(args.data))[0]
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_csv = f"data/log/trades_{data_base}_{timestamp}.csv"
    save_trades_csv(res["trades"], out_csv, extra_cols={
        "strategy": name,
        "data_file": args.data,
        "fee": args.fee,
        "slippage": args.slippage,
        "initial_cash": args.initial_cash,
    })
    print(f"Trade log saved: {out_csv}")

    print_summary(res)

    if args.plot or args.save_chart:
        out_png = args.save_chart
        plot_results(df, res, data_name=args.data, show=args.plot, save_path=out_png)

if __name__ == "__main__":
    main()
