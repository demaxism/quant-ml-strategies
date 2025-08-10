import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch

# Customize the test start shift (number of fine_df bars to skip at the start of test)
# Just use to check if it has impact on results
START_SHIFT = 0

# === Backtest date range control ===
# Set BT_FROM and BT_UNTIL to limit the backtest to a specific date range in fine_df.
# Use None to disable either bound. Format should match fine_df's index (e.g., '2022-01-01' or pd.Timestamp).
# BT_FROM = '2025-03-01'   # e.g., '2022-01-01', default None
# BT_UNTIL = None  # e.g., '2023-01-01', default None

def backtest_realtime_lstm(
    model, df, split, SEQ_LEN, PREDICT_AHEAD, N_HOLD, timestamp, scaler, WRITE_CSV=False, REVERT_PROFIT=False, threshold=0.008, allowance=0.002, symbol=None, fine_df=None, BT_FROM=None, BT_UNTIL=None, test_start=None
):
    """
    Simulate real-time trading: for each new fine_df bar, update position and equity.
    Model predictions are made only when a new df bar is complete (aggregated from fine_df).
    model: trained LSTM model (in eval mode)
    df: pandas DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'change']
    fine_df: higher-resolution DataFrame (e.g., 1h if df is 4h)
    split: int, train/test split index (on df)
    SEQ_LEN: int, sequence length used in LSTM
    PREDICT_AHEAD: int, how many bars ahead to predict
    scaler: fitted MinMaxScaler (for normalization)
    timestamp: str, for file naming
    """
    import csv

    if fine_df is None:
        # Fallback to old behavior if fine_df not provided
        # (old code block here, omitted for brevity)
        raise ValueError("fine_df must be provided for fine-grained backtest.")

    # === Filter fine_df by BT_FROM and BT_UNTIL if set ===
    if BT_FROM is not None and BT_UNTIL is not None:
        fine_df = fine_df.loc[BT_FROM:BT_UNTIL].copy()
    elif BT_FROM is not None:
        fine_df = fine_df.loc[BT_FROM:].copy()
    elif BT_UNTIL is not None:
        fine_df = fine_df.loc[:BT_UNTIL].copy()

    # === Add technical indicators to fine_df ===
    # MA30 and diff
    fine_df['ma30'] = fine_df['close'].rolling(window=30).mean()
    fine_df['diff_ma30'] = fine_df['ma30'].diff()
    # MA10 and diff
    fine_df['ma10'] = fine_df['close'].rolling(window=10).mean()
    fine_df['diff_ma10'] = fine_df['ma10'].diff()
    # RSI (14)
    delta = fine_df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    fine_df['rsi'] = 100 - (100 / (1 + rs))
    # Bollinger Bands (20, 2)
    fine_df['bb_middle'] = fine_df['close'].rolling(window=20).mean()
    fine_df['bb_std'] = fine_df['close'].rolling(window=20).std()
    fine_df['bb_upper'] = fine_df['bb_middle'] + 2 * fine_df['bb_std']
    fine_df['bb_lower'] = fine_df['bb_middle'] - 2 * fine_df['bb_std']

    # Determine how many fine_df bars per df bar
    # Assume df and fine_df have the same symbol, and are continuous
    # Use the time difference between first two rows to infer frequency
    df_freq = (df.index[1] - df.index[0]).total_seconds()
    fine_freq = (fine_df.index[1] - fine_df.index[0]).total_seconds()
    bars_per_df = int(round(df_freq / fine_freq))

    # Prepare test data
    # Determine test_start based on BT_FROM
    _test_start = BT_FROM if BT_FROM is not None else test_start
    if _test_start is not None:
        loc = fine_df.index.get_loc(_test_start)
        if isinstance(loc, slice):
            test_start = loc.start
        elif isinstance(loc, (np.integer, int)):
            test_start = int(loc)
        else:
            raise ValueError(f"Unexpected result from fine_df.index.get_loc(BT_FROM): {loc}")
    else:
        raise ValueError(f"No valid test_start provided. Please set BT_FROM or test_start.")
    test_start += START_SHIFT  # apply the start shift
    test_end = len(fine_df) - PREDICT_AHEAD * bars_per_df  # enough fine bars for SEQ_LEN+PREDICT_AHEAD df bars

    print(f"test_start: {test_start}, test_end: {test_end}, fine_df length: {len(fine_df)}   ")
    equity = [1.0]  # start with $1
    position = 0    # 0 = flat, 1 = long
    entry_price = 0
    entry_date = None
    entry_take_profit = None
    entry_stop_loss = None
    bars_held = 0  # Number of fine_df bars position has been held
    trade_log = []
    last_equity = equity[-1]
    current_stop_loss = None
    detailed_log = []
    prev_take_profit_price = None
    prev_stop_loss_price = None

    # For plotting and stats
    equity_dates = []
    close_prices = []
    open_prices = []
    high_prices = []
    low_prices = []
    volumes = []

    # Rolling window of df bars (for model input)
    df_window = df.iloc[:split+SEQ_LEN].copy()
    # If not enough, pad with earlier bars
    if len(df_window) < SEQ_LEN:
        raise ValueError("Not enough df bars for SEQ_LEN window at split point.")

    # State for aggregation
    fine_buffer = []
    last_model_pred = None
    last_df_bar = None
    last_df_bar_idx = split + SEQ_LEN - 1  # index in df

    # Helper for trade log
    def log_trade(entry_date, entry_price, exit_date, exit_price, pnl, exit_type):
        trade_log.append({
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_type': exit_type
        })

    # Helper for detailed log
    def log_detailed(dt, o, h, l, c, v, state, entry_price, take_profit, stop_loss, exit_method, pnl, abs_pnl, pnl_detail="",
                     ma30=None, diff_ma30=None, ma10=None, diff_ma10=None, rsi=None, bb_upper=None, bb_middle=None, bb_lower=None, memo=""):
        entry = {
            'datetime': dt,
            'open': o,
            'high': h,
            'low': l,
            'close': c,
            'volume': v,
            'state': state,
            'entry_price': entry_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'exit_method': exit_method,
            'pnl': pnl,
            'abs_pnl': abs_pnl,
            'pnl_detail': pnl_detail,
            'ma30': ma30,
            'diff_ma30': diff_ma30,
            'ma10': ma10,
            'diff_ma10': diff_ma10,
            'rsi': rsi,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'memo': memo
        }
        detailed_log.append(entry)
        # If this is an exit with take_profit or stop_loss, annotate the entry
        if state == "exit" and exit_method in ("take_profit", "stop_loss"):
            # Find the most recent entry with matching entry_date and entry_price
            for prev in reversed(detailed_log):
                if prev['state'] == "entry" and prev['entry_price'] == entry_price and prev['datetime'] == entry_date:
                    if exit_method == "take_profit":
                        prev['memo'] = "will_win"
                    elif exit_method == "stop_loss":
                        prev['memo'] = "will_lose"
                    break

    # Main loop: iterate over fine_df bars in test set
    for i in range(test_start, test_end):
        fine_row = fine_df.iloc[i]
        fine_date = fine_row.name
        fine_open = fine_row['open']
        fine_high = fine_row['high']
        fine_low = fine_row['low']
        fine_close = fine_row['close']
        fine_volume = fine_row['volume']
        fine_change = fine_row['change']

        fine_buffer.append(fine_row)

        is_boundary = False

        # Check if a new df bar is complete (every bars_per_df fine bars)
        if len(fine_buffer) == bars_per_df:
            is_boundary = True
            # Aggregate fine_buffer into a new df bar
            agg_open = fine_buffer[0]['open']
            agg_high = max(row['high'] for row in fine_buffer)
            agg_low = min(row['low'] for row in fine_buffer)
            agg_close = fine_buffer[-1]['close']
            agg_volume = sum(row['volume'] for row in fine_buffer)
            agg_change = fine_buffer[-1]['change']
            agg_date = fine_buffer[-1].name  # use last fine bar's datetime as df bar's datetime

            new_df_bar = {
                'open': agg_open,
                'high': agg_high,
                'low': agg_low,
                'close': agg_close,
                'volume': agg_volume,
                'change': agg_change
            }
            # Append to df_window (rolling)
            df_window = pd.concat([df_window, pd.DataFrame([new_df_bar], index=[agg_date])])
            if len(df_window) > SEQ_LEN:
                df_window = df_window.iloc[-SEQ_LEN:]

            # Prepare input sequence for model
            seq = df_window
            seq_scaled = scaler.transform(seq)
            x = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)  # shape (1, SEQ_LEN, features)
            model.eval()
            with torch.no_grad():
                pred = model(x).numpy()[0]  # shape (2,): [high_return, low_return]
            last_model_pred = pred
            last_df_bar = new_df_bar
            last_df_bar_idx += 1

            fine_buffer = []  # reset buffer

        # Use last_model_pred for trading logic
        if last_model_pred is not None:
            curr_close = fine_close
            curr_open = fine_open
            curr_high = fine_high
            curr_low = fine_low
            curr_volume = fine_volume
            curr_date = fine_date

            # Get indicator values for this bar
            ma30 = fine_row.get('ma30', None)
            diff_ma30 = fine_row.get('diff_ma30', None)
            ma10 = fine_row.get('ma10', None)
            diff_ma10 = fine_row.get('diff_ma10', None)
            rsi = fine_row.get('rsi', None)
            bb_upper = fine_row.get('bb_upper', None)
            bb_middle = fine_row.get('bb_middle', None)
            bb_lower = fine_row.get('bb_lower', None)

            # Reconstruct predicted high/low prices from last df bar's close
            pred_high = last_df_bar['close'] * (1 + last_model_pred[0])
            pred_low = last_df_bar['close'] * (1 + last_model_pred[1])

            exited_this_bar = False
            take_profit_price = pred_high - curr_close * allowance
            stop_loss_price = min(pred_low - curr_close * allowance, curr_close - (take_profit_price - curr_close) * 0.5) # Ensure stop loss is below current close

            # For plotting/stats
            equity_dates.append(curr_date)
            close_prices.append(curr_close)
            open_prices.append(curr_open)
            high_prices.append(curr_high)
            low_prices.append(curr_low)
            volumes.append(curr_volume)

            # Exit logic: only if in position
            if position == 1:
                bars_held += 1

                # Always update prev_take_profit_price and prev_stop_loss_price for use in the next bar
                if bars_held >= 1 and bars_held <= 2:
                    # Take profit hit
                    if curr_high >= prev_take_profit_price:
                        if curr_open > prev_take_profit_price:
                            exit_price = curr_open
                            pnl_detail = f"({curr_open:.6f} - {entry_price:.6f}) / {entry_price:.6f}"
                        else:
                            exit_price = prev_take_profit_price
                            pnl_detail = f"({prev_take_profit_price:.6f} - {entry_price:.6f}) / {entry_price:.6f}"
                        pnl = (exit_price - entry_price) / entry_price
                        pnl = -pnl if REVERT_PROFIT else pnl
                        last_equity = last_equity * (1 + pnl)
                        log_trade(entry_date, entry_price, curr_date, prev_take_profit_price, pnl, "take_profit")
                        pnl_detail = f"({prev_take_profit_price:.6f} - {entry_price:.6f}) / {entry_price:.6f}"
                        log_detailed(
                            curr_date, curr_open, curr_high, curr_low, curr_close, curr_volume,
                            "exit", entry_price, prev_take_profit_price, current_stop_loss, "take_profit", pnl, last_equity - (last_equity / (1 + pnl)), pnl_detail,
                            ma30, diff_ma30, ma10, diff_ma10, rsi, bb_upper, bb_middle, bb_lower
                        )
                        position = 0
                        bars_held = 0
                        exited_this_bar = True
                        current_stop_loss = None
                        prev_take_profit_price = None
                        prev_stop_loss_price = None

                    # For the first two bars, we set TP/SL directly from the prediction
                    prev_take_profit_price = take_profit_price
                    prev_stop_loss_price = stop_loss_price
                else:
                    if current_stop_loss is None:
                        current_stop_loss = prev_stop_loss_price
                    else:
                        if prev_stop_loss_price > current_stop_loss:
                            current_stop_loss = prev_stop_loss_price

                    # Always update take profit price per bar
                    prev_take_profit_price = take_profit_price

                    # Max hold condition
                    if bars_held >= N_HOLD * bars_per_df:
                        pnl = (curr_close - entry_price) / entry_price
                        pnl = -pnl if REVERT_PROFIT else pnl
                        last_equity = last_equity * (1 + pnl)
                        log_trade(entry_date, entry_price, curr_date, curr_close, pnl, "max_hold")
                        pnl_detail = f"({curr_close:.6f} - {entry_price:.6f}) / {entry_price:.6f}"
                        log_detailed(
                            curr_date, curr_open, curr_high, curr_low, curr_close, curr_volume,
                            "exit", entry_price, prev_take_profit_price, current_stop_loss, "max_hold", pnl, last_equity - (last_equity / (1 + pnl)), pnl_detail,
                            ma30, diff_ma30, ma10, diff_ma10, rsi, bb_upper, bb_middle, bb_lower
                        )
                        position = 0
                        bars_held = 0
                        exited_this_bar = True
                        current_stop_loss = None
                        prev_take_profit_price = None
                        prev_stop_loss_price = None
                    elif curr_high >= prev_take_profit_price and curr_low <= current_stop_loss:
                        # Both TP and SL penetrated
                        mid = (curr_high + curr_low) / 2
                        dist_to_tp = abs(prev_take_profit_price - mid)
                        dist_to_sl = abs(current_stop_loss - mid)
                        if dist_to_tp < dist_to_sl:
                            exit_price = prev_take_profit_price
                        else:
                            exit_price = current_stop_loss
                        pnl = (exit_price - entry_price) / entry_price
                        pnl = -pnl if REVERT_PROFIT else pnl
                        last_equity = last_equity * (1 + pnl)
                        log_trade(entry_date, entry_price, curr_date, exit_price, pnl, "both_penetrate")
                        pnl_detail = f"({exit_price:.6f} - {entry_price:.6f}) / {entry_price:.6f}"
                        log_detailed(
                            curr_date, curr_open, curr_high, curr_low, curr_close, curr_volume,
                            "exit", entry_price, prev_take_profit_price, current_stop_loss, "both_penetrate", pnl, last_equity - (last_equity / (1 + pnl)), pnl_detail,
                            ma30, diff_ma30, ma10, diff_ma10, rsi, bb_upper, bb_middle, bb_lower
                        )
                        position = 0
                        bars_held = 0
                        exited_this_bar = True
                        current_stop_loss = None
                        prev_take_profit_price = None
                        prev_stop_loss_price = None
                    elif curr_low <= current_stop_loss:
                        # Stop loss hit
                        if curr_open > current_stop_loss:
                            exit_price = current_stop_loss
                        else:
                            # If open is below stop loss, wait for it hitting stop loss to exit, otherwise exit at close
                            if curr_high >= current_stop_loss:
                                exit_price = current_stop_loss
                            else:
                                exit_price = curr_close
                        pnl_detail = f"({exit_price:.6f} - {entry_price:.6f}) / {entry_price:.6f}"
                        pnl = (exit_price - entry_price) / entry_price
                        pnl = -pnl if REVERT_PROFIT else pnl
                        last_equity = last_equity * (1 + pnl)
                        log_trade(entry_date, entry_price, curr_date, exit_price, pnl, "stop_loss")
                        log_detailed(
                            curr_date, curr_open, curr_high, curr_low, curr_close, curr_volume,
                            "exit", entry_price, prev_take_profit_price, exit_price, "stop_loss", pnl, last_equity - (last_equity / (1 + pnl)), pnl_detail,
                            ma30, diff_ma30, ma10, diff_ma10, rsi, bb_upper, bb_middle, bb_lower
                        )
                        position = 0
                        bars_held = 0
                        exited_this_bar = True
                        current_stop_loss = None
                        prev_take_profit_price = None
                        prev_stop_loss_price = None
                    elif curr_high >= prev_take_profit_price:
                        # Take profit hit
                        if curr_open > prev_take_profit_price:
                            exit_price = curr_open
                            pnl_detail = f"({curr_open:.6f} - {entry_price:.6f}) / {entry_price:.6f}"
                        else:
                            exit_price = prev_take_profit_price
                            pnl_detail = f"({prev_take_profit_price:.6f} - {entry_price:.6f}) / {entry_price:.6f}"
                        pnl = (exit_price - entry_price) / entry_price
                        pnl = -pnl if REVERT_PROFIT else pnl
                        last_equity = last_equity * (1 + pnl)
                        log_trade(entry_date, entry_price, curr_date, prev_take_profit_price, pnl, "take_profit")
                        pnl_detail = f"({prev_take_profit_price:.6f} - {entry_price:.6f}) / {entry_price:.6f}"
                        log_detailed(
                            curr_date, curr_open, curr_high, curr_low, curr_close, curr_volume,
                            "exit", entry_price, prev_take_profit_price, current_stop_loss, "take_profit", pnl, last_equity - (last_equity / (1 + pnl)), pnl_detail,
                            ma30, diff_ma30, ma10, diff_ma10, rsi, bb_upper, bb_middle, bb_lower
                        )
                        position = 0
                        bars_held = 0
                        exited_this_bar = True
                        current_stop_loss = None
                        prev_take_profit_price = None
                        prev_stop_loss_price = None
                    else:
                        prev_take_profit_price = take_profit_price
                        prev_stop_loss_price = stop_loss_price

                if exited_this_bar:
                    prev_take_profit_price = None
                    prev_stop_loss_price = None

            # Entry logic: only if not in position and did not just exit
            if position == 0 and not exited_this_bar and take_profit_price > curr_close * (1 + threshold) and is_boundary:
                if stop_loss_price < curr_close:
                    position = 1
                    entry_price = curr_close
                    entry_date = curr_date
                    entry_take_profit = take_profit_price
                    entry_stop_loss = stop_loss_price
                    bars_held = 1
                    current_stop_loss = stop_loss_price
                    prev_take_profit_price = take_profit_price
                    prev_stop_loss_price = stop_loss_price
                    log_detailed(
                        curr_date, curr_open, curr_high, curr_low, curr_close, curr_volume,
                        "entry", entry_price, entry_take_profit, entry_stop_loss, "", 0, 0, "",
                        ma30, diff_ma30, ma10, diff_ma10, rsi, bb_upper, bb_middle, bb_lower
                    )
                else:
                    log_detailed(
                        curr_date, curr_open, curr_high, curr_low, curr_close, curr_volume,
                        "skip_entry", curr_close, take_profit_price, stop_loss_price, "invalid_tp_sl", 0, 0, "",
                        ma30, diff_ma30, ma10, diff_ma10, rsi, bb_upper, bb_middle, bb_lower
                    )
            elif position == 1 and not exited_this_bar:
                log_detailed(
                    curr_date, curr_open, curr_high, curr_low, curr_close, curr_volume,
                    "holding", entry_price, prev_take_profit_price, current_stop_loss, "", 0, 0, f"bars_held={bars_held}",
                    ma30, diff_ma30, ma10, diff_ma10, rsi, bb_upper, bb_middle, bb_lower
                )

            equity.append(last_equity)

    # If still in position at the end, close at last close
    if position == 1:
        pnl = (close_prices[-1] - entry_price) / entry_price
        pnl = -pnl if REVERT_PROFIT else pnl
        last_equity = last_equity * (1 + pnl)
        equity.append(last_equity)
        log_trade(entry_date, entry_price, equity_dates[-1], close_prices[-1], pnl, "final_close")
        pnl_detail = f"({close_prices[-1]:.6f} - {entry_price:.6f}) / {entry_price:.6f}"
        # For the final close, get indicators from the last fine_df row
        last_idx = fine_df.index.get_loc(equity_dates[-1]) if equity_dates[-1] in fine_df.index else -1
        if last_idx != -1:
            last_row = fine_df.iloc[last_idx]
            ma30 = last_row.get('ma30', None)
            diff_ma30 = last_row.get('diff_ma30', None)
            ma10 = last_row.get('ma10', None)
            diff_ma10 = last_row.get('diff_ma10', None)
            rsi = last_row.get('rsi', None)
            bb_upper = last_row.get('bb_upper', None)
            bb_middle = last_row.get('bb_middle', None)
            bb_lower = last_row.get('bb_lower', None)
        else:
            ma30 = diff_ma30 = ma10 = diff_ma10 = rsi = bb_upper = bb_middle = bb_lower = None

        log_detailed(
            equity_dates[-1], open_prices[-1], high_prices[-1], low_prices[-1], close_prices[-1], volumes[-1],
            "exit", entry_price, entry_take_profit, entry_stop_loss, "final_close", pnl, last_equity - (last_equity / (1 + pnl)), pnl_detail,
            ma30, diff_ma30, ma10, diff_ma10, rsi, bb_upper, bb_middle, bb_lower
        )
    else:
        equity.append(last_equity)

    # Write detailed log to CSV
    if WRITE_CSV:
        log_filename = f"data/log/lstm_backtest_realtime_log_{timestamp}.csv"
        log_columns = [
            'datetime', 'open', 'high', 'low', 'close', 'volume',
            'state', 'entry_price', 'take_profit', 'stop_loss', 'exit_method', 'pnl', 'abs_pnl', 'pnl_detail',
            'ma30', 'diff_ma30', 'ma10', 'diff_ma10', 'rsi', 'bb_upper', 'bb_middle', 'bb_lower', 'memo'
        ]
        with open(log_filename, 'w', newline='') as csvfile:
            import csv
            writer = csv.DictWriter(csvfile, fieldnames=log_columns)
            writer.writeheader()
            for row in detailed_log:
                writer.writerow(row)
        print(f"Detailed backtest log written to {log_filename}")

    # Compute stats
    returns = np.array([t['pnl'] for t in trade_log])
    total_return = equity[-1] - 1.0
    number_of_trades = len(trade_log)
    win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
    max_drawdown = np.max(np.maximum.accumulate(equity[:-1]) - equity[:-1])

    if WRITE_CSV:
        # Plot equity curve with price overlay (separate Y-axes)
        plt.figure(figsize=(12, 5))
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        print(f"equity_dates length: {len(equity_dates)}, equity length: {len(equity)}, close_prices length: {len(close_prices)}")
        min_len = min(len(equity_dates), len(equity), len(close_prices))
        plot_dates = equity_dates[:min_len]
        plot_equity = equity[:min_len]

        l1, = ax1.plot(plot_dates, plot_equity, label='Equity Curve', color='blue')
        ax1.set_ylabel('Equity ($)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        l2, = ax2.plot(plot_dates, close_prices[:min_len], label='Price', color='orange', alpha=0.5)
        ax2.set_ylabel('Price', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        plt.title(f'Equity Curve {symbol} (Realtime Sim) with Price Overlay')
        ax1.set_xlabel('Date')
        ax1.grid(True)

        lines = [l1, l2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.tight_layout()
        plt.savefig(f"data/log/lstm_equity_curve_realtime_{symbol}_{timestamp}.png")

    return trade_log, equity, total_return, number_of_trades, win_rate, max_drawdown
