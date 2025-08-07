import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def backtest_realtime_lstm(
    model, df, split, SEQ_LEN, PREDICT_AHEAD, N_HOLD, timestamp, scaler, WRITE_CSV=False, REVERT_PROFIT=False, threshold=0.008, allowance=0.002, symbol=None
):
    """
    Simulate real-time trading: for each new bar in the test set, use the model to predict the future,
    and make trading decisions bar-by-bar (no batch prediction).
    model: trained LSTM model (in eval mode)
    df: pandas DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'change']
    split: int, train/test split index
    SEQ_LEN: int, sequence length used in LSTM
    PREDICT_AHEAD: int, how many bars ahead to predict
    scaler: fitted MinMaxScaler (for normalization)
    timestamp: str, for file naming
    """
    import csv

    # Prepare test data
    df_values = df.values  # original (unscaled) values
    test_start = split
    test_end = len(df) - PREDICT_AHEAD  # so we have enough bars for SEQ_LEN + PREDICT_AHEAD

    equity = [1.0]  # start with $1
    position = 0    # 0 = flat, 1 = long
    entry_price = 0
    entry_date = None
    entry_take_profit = None
    entry_stop_loss = None
    bars_held = 0  # Number of bars position has been held
    trade_log = []
    last_equity = equity[-1]
    current_stop_loss = None
    detailed_log = []
    prev_take_profit_price = None
    prev_stop_loss_price = None

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
    def log_detailed(dt, o, h, l, c, v, state, entry_price, take_profit, stop_loss, exit_method, pnl, abs_pnl, pnl_detail=""):
        detailed_log.append({
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
            'pnl_detail': pnl_detail
        })

    # For plotting and stats
    equity_dates = []
    close_prices = []
    open_prices = []
    high_prices = []
    low_prices = []
    volumes = []

    # Main loop: simulate receiving each new bar in the test set
    for i in range(test_start, test_end - SEQ_LEN - PREDICT_AHEAD + 2):
        # i is the index of the first bar in the SEQ_LEN window
        seq_idx = i
        # Prepare input sequence: SEQ_LEN bars ending at seq_idx + SEQ_LEN - 1
        seq = df.iloc[seq_idx:seq_idx+SEQ_LEN]
        seq_scaled = scaler.transform(seq)
        x = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)  # shape (1, SEQ_LEN, features)
        model.eval()
        with torch.no_grad():
            pred = model(x).numpy()[0]  # shape (2,): [high_return, low_return]

        # Get current bar info (the last bar in the sequence)
        curr_idx = seq_idx + SEQ_LEN - 1
        curr_row = df.iloc[curr_idx]
        curr_close = curr_row['close']
        curr_open = curr_row['open']
        curr_high = curr_row['high']
        curr_low = curr_row['low']
        curr_volume = curr_row['volume']
        curr_date = curr_row.name  # index is datetime

        # Reconstruct predicted high/low prices
        pred_high = curr_close * (1 + pred[0])
        pred_low = curr_close * (1 + pred[1])

        exited_this_bar = False
        take_profit_price = pred_high * (1 - allowance)
        stop_loss_price = pred_low * (1 + allowance)

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
            if bars_held == 1:
                # On entry bar, store current TP/SL for use in t1
                prev_take_profit_price = take_profit_price
                prev_stop_loss_price = stop_loss_price
            elif bars_held == 2:
                # On t1, update prev TP/SL for use in t2, but do not check TP/SL
                prev_take_profit_price = take_profit_price
                prev_stop_loss_price = stop_loss_price
            else:
                # bars_held >= 3: check TP/SL using previous bar's TP/SL prices
                # Trailing stop: only update stop_loss if it increases
                if current_stop_loss is None:
                    current_stop_loss = prev_stop_loss_price
                else:
                    if prev_stop_loss_price > current_stop_loss:
                        current_stop_loss = prev_stop_loss_price

                # Hold up to N_HOLD bars, then exit at current close
                if bars_held >= N_HOLD:
                    pnl = (curr_close - entry_price) / entry_price
                    pnl = -pnl if REVERT_PROFIT else pnl
                    last_equity = last_equity * (1 + pnl)
                    log_trade(entry_date, entry_price, curr_date, curr_close, pnl, "max_hold")
                    pnl_detail = f"({curr_close:.6f} - {entry_price:.6f}) / {entry_price:.6f}"
                    log_detailed(
                        curr_date, curr_open, curr_high, curr_low, curr_close, curr_volume,
                        "exit", entry_price, prev_take_profit_price, current_stop_loss, "max_hold", pnl, last_equity - (last_equity / (1 + pnl)), pnl_detail
                    )
                    position = 0
                    bars_held = 0
                    exited_this_bar = True
                    current_stop_loss = None
                    prev_take_profit_price = None
                    prev_stop_loss_price = None
                # 0. Bar penetrate both TP/SL
                elif curr_high >= prev_take_profit_price and curr_low <= current_stop_loss:
                    # Take the median of the two prices as exit price
                    exit_price = (prev_take_profit_price + current_stop_loss) / 2
                    pnl = (exit_price - entry_price) / entry_price
                    pnl = -pnl if REVERT_PROFIT else pnl
                    last_equity = last_equity * (1 + pnl)
                    log_trade(entry_date, entry_price, curr_date, exit_price, pnl, "both_penetrate")
                    pnl_detail = f"({exit_price:.6f} - {entry_price:.6f}) / {entry_price:.6f}"
                    log_detailed(
                        curr_date, curr_open, curr_high, curr_low, curr_close, curr_volume,
                        "exit", entry_price, prev_take_profit_price, current_stop_loss, "both_penetrate", pnl, last_equity - (last_equity / (1 + pnl)), pnl_detail
                    )
                    # Reset for next trade
                    position = 0
                    bars_held = 0
                    exited_this_bar = True
                    current_stop_loss = None
                    prev_take_profit_price = None
                    prev_stop_loss_price = None
                # 1. Stop loss (use trailing current_stop_loss)
                elif curr_low <= current_stop_loss:
                    # If current open is below stop loss, exit at open
                    if curr_open > current_stop_loss:
                        exit_price = current_stop_loss
                    else:
                        exit_price = curr_open
                    pnl_detail = f"({exit_price:.6f} - {entry_price:.6f}) / {entry_price:.6f}"
                
                    pnl = (exit_price - entry_price) / entry_price
                    pnl = -pnl if REVERT_PROFIT else pnl
                    last_equity = last_equity * (1 + pnl)
                    log_trade(entry_date, entry_price, curr_date, exit_price, pnl, "stop_loss")
                    log_detailed(
                        curr_date, curr_open, curr_high, curr_low, curr_close, curr_volume,
                        "exit", entry_price, prev_take_profit_price, exit_price, "stop_loss", pnl, last_equity - (last_equity / (1 + pnl)), pnl_detail
                    )
                    position = 0
                    bars_held = 0
                    exited_this_bar = True
                    current_stop_loss = None
                    prev_take_profit_price = None
                    prev_stop_loss_price = None
                # 2. Take profit (use previous bar's take_profit_price)
                elif curr_high >= prev_take_profit_price:
                    # If current open is above take profit, exit at open
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
                        "exit", entry_price, prev_take_profit_price, current_stop_loss, "take_profit", pnl, last_equity - (last_equity / (1 + pnl)), pnl_detail
                    )
                    position = 0
                    bars_held = 0
                    exited_this_bar = True
                    current_stop_loss = None
                    prev_take_profit_price = None
                    prev_stop_loss_price = None
                else:
                    # After check, update prev TP/SL for next bar
                    prev_take_profit_price = take_profit_price
                    prev_stop_loss_price = stop_loss_price

            # If exited, reset prev TP/SL
            if exited_this_bar:
                prev_take_profit_price = None
                prev_stop_loss_price = None

        # Entry logic: only if not in position and did not just exit
        if position == 0 and not exited_this_bar and pred_high > curr_close * (1 + threshold):
            if (take_profit_price > curr_close) and (stop_loss_price < curr_close):
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
                    "entry", entry_price, entry_take_profit, entry_stop_loss, "", 0, 0, ""
                )
            else:
                log_detailed(
                    curr_date, curr_open, curr_high, curr_low, curr_close, curr_volume,
                    "skip_entry", curr_close, take_profit_price, stop_loss_price, "invalid_tp_sl", 0, 0, ""
                )
        elif position == 1 and not exited_this_bar:
            log_detailed(
                curr_date, curr_open, curr_high, curr_low, curr_close, curr_volume,
                "holding", entry_price, entry_take_profit, entry_stop_loss, "", 0, 0, f"bars_held={bars_held}"
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
        log_detailed(
            equity_dates[-1], open_prices[-1], high_prices[-1], low_prices[-1], close_prices[-1], volumes[-1],
            "exit", entry_price, entry_take_profit, entry_stop_loss, "final_close", pnl, last_equity - (last_equity / (1 + pnl)), pnl_detail
        )
    else:
        equity.append(last_equity)

    # Write detailed log to CSV
    if WRITE_CSV:
        log_filename = f"data/lstm_backtest_realtime_log_{timestamp}.csv"
        log_columns = [
            'datetime', 'open', 'high', 'low', 'close', 'volume',
            'state', 'entry_price', 'take_profit', 'stop_loss', 'exit_method', 'pnl', 'abs_pnl', 'pnl_detail'
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

    # Plot equity curve with price overlay (separate Y-axes)
    plt.figure(figsize=(12, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # print the length of equity_dates, equity, close_prices
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
    plt.savefig(f"data/lstm_equity_curve_realtime_{symbol}_{timestamp}.png")

    return trade_log, equity, total_return, number_of_trades, win_rate, max_drawdown
