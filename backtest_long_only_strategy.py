import numpy as np
import matplotlib.pyplot as plt
import os

def backtest_long_only_strategy(
    true, predicted, date_index, df, split, SEQ_LEN, N_HOLD, timestamp, WRITE_CSV=False, REVERT_PROFIT=False, threshold=0.008, allowance=0.002, symbol=None
):
    """
    true: [N, 2] array of true [high, low] (not used here), but we use close prices from df
    predicted: [N, 2] array of predicted [high, low]
    date_index: DatetimeIndex for test set
    df: pandas DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
    split: int, train/test split index
    SEQ_LEN: int, sequence length used in LSTM
    timestamp: str, for file naming
    WRITE_CSV: bool, whether to write detailed log to CSV
    threshold: float, e.g. 0.002 for 0.2%; entry condition: pred_high > curr_close * (1 + threshold)
    allowance: float, take profit/stop loss trigger as a fraction (default 0.002 = 0.2%)
    """
    import csv

    close_prices = df['close'].values[SEQ_LEN + split + 1:]
    close_prices = close_prices[:len(predicted)]
    high_prices = df['high'].values[SEQ_LEN + split + 1:]
    high_prices = high_prices[:len(predicted)]
    low_prices = df['low'].values[SEQ_LEN + split + 1:]
    low_prices = low_prices[:len(predicted)]
    open_prices = df['open'].values[SEQ_LEN + split + 1:]
    open_prices = open_prices[:len(predicted)]
    volumes = df['volume'].values[SEQ_LEN + split + 1:]
    volumes = volumes[:len(predicted)]
    dates = date_index[:len(predicted)]

    equity = [1.0]  # start with $1
    position = 0    # 0 = flat, 1 = long
    entry_price = 0
    entry_date = None
    entry_take_profit = None
    entry_stop_loss = None
    bars_held = 0  # Number of bars position has been held
    trade_log = []
    last_equity = equity[-1]

    # For trailing stop logic
    current_stop_loss = None

    detailed_log = []

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

    min_len = min(
        len(predicted),
        len(close_prices),
        len(high_prices),
        len(low_prices),
        len(open_prices),
        len(volumes),
        len(dates)
    )
    for i in range(min_len - 1):  # last prediction can't be traded (no next close)
        curr_close = close_prices[i]
        curr_open = open_prices[i]
        curr_high = high_prices[i]
        curr_low = low_prices[i]
        curr_volume = volumes[i]
        curr_date = dates[i]
        next_close = close_prices[i+1]
        next_high = high_prices[i+1]
        next_low = low_prices[i+1]
        pred_high = predicted[i, 0]
        pred_low = predicted[i, 1]
        exited_this_bar = False

        # Default state
        state = "flat"
        exit_method = ""
        # Dynamically update take profit and stop loss for each holding bar
        take_profit_price = pred_high * (1 - allowance)
        stop_loss_price = pred_low * (1 + allowance)

        # Exit logic: only if in position
        if position == 1:
            bars_held += 1

            # Trailing stop: only update stop_loss if it increases
            if current_stop_loss is None:
                current_stop_loss = stop_loss_price
            else:
                if stop_loss_price > current_stop_loss:
                    current_stop_loss = stop_loss_price

            # 1. Stop loss (use trailing current_stop_loss)
            if next_low <= current_stop_loss:
                pnl = (current_stop_loss - entry_price) / entry_price
                pnl = -pnl if REVERT_PROFIT else pnl
                last_equity = last_equity * (1 + pnl)
                equity.append(last_equity)
                log_trade(entry_date, entry_price, dates[i+1], current_stop_loss, pnl, "stop_loss")
                pnl_detail = f"({current_stop_loss:.6f} - {entry_price:.6f}) / {entry_price:.6f}"
                log_detailed(
                    dates[i+1], open_prices[i+1], high_prices[i+1], low_prices[i+1], close_prices[i+1], volumes[i+1],
                    "exit", entry_price, take_profit_price, current_stop_loss, "stop_loss", pnl, last_equity - (last_equity / (1 + pnl)), pnl_detail
                )
                position = 0
                bars_held = 0
                exited_this_bar = True
                current_stop_loss = None
            # 2. Take profit (use current bar's recalculated take_profit_price)
            elif next_high >= take_profit_price:
                pnl = (take_profit_price - entry_price) / entry_price
                pnl = -pnl if REVERT_PROFIT else pnl
                last_equity = last_equity * (1 + pnl)
                equity.append(last_equity)
                log_trade(entry_date, entry_price, dates[i+1], take_profit_price, pnl, "take_profit")
                pnl_detail = f"({take_profit_price:.6f} - {entry_price:.6f}) / {entry_price:.6f}"
                log_detailed(
                    dates[i+1], open_prices[i+1], high_prices[i+1], low_prices[i+1], close_prices[i+1], volumes[i+1],
                    "exit", entry_price, take_profit_price, current_stop_loss, "take_profit", pnl, last_equity - (last_equity / (1 + pnl)), pnl_detail
                )
                position = 0
                bars_held = 0
                exited_this_bar = True
                current_stop_loss = None
            # 3. Hold up to N_HOLD bars, then exit at next close
            elif bars_held >= N_HOLD:
                pnl = (next_close - entry_price) / entry_price
                pnl = -pnl if REVERT_PROFIT else pnl
                last_equity = last_equity * (1 + pnl)
                equity.append(last_equity)
                log_trade(entry_date, entry_price, dates[i+1], next_close, pnl, "max_hold")
                pnl_detail = f"({next_close:.6f} - {entry_price:.6f}) / {entry_price:.6f}"
                log_detailed(
                    dates[i+1], open_prices[i+1], high_prices[i+1], low_prices[i+1], close_prices[i+1], volumes[i+1],
                    "exit", entry_price, take_profit_price, current_stop_loss, "max_hold", pnl, last_equity - (last_equity / (1 + pnl)), pnl_detail
                )
                position = 0
                bars_held = 0
                exited_this_bar = True
                current_stop_loss = None

        # Entry logic: only if not in position and did not just exit
        if position == 0 and not exited_this_bar and pred_high > curr_close * (1 + threshold):
            # Only enter if take profit is above entry and stop loss is below entry
            if (take_profit_price > curr_close) and (stop_loss_price < curr_close):
                position = 1
                entry_price = curr_close
                entry_date = curr_date
                entry_take_profit = take_profit_price
                entry_stop_loss = stop_loss_price
                bars_held = 1
                current_stop_loss = stop_loss_price  # Initialize trailing stop at entry
                log_detailed(
                    curr_date, curr_open, curr_high, curr_low, curr_close, curr_volume,
                    "entry", entry_price, entry_take_profit, entry_stop_loss, "", 0, 0, ""
                )
            else:
                # Skip entry if invalid take profit/stop loss
                log_detailed(
                    curr_date, curr_open, curr_high, curr_low, curr_close, curr_volume,
                    "skip_entry", curr_close, take_profit_price, stop_loss_price, "invalid_tp_sl", 0, 0, ""
                )
        elif position == 1 and not exited_this_bar:
            # Log holding bar
            log_detailed(
                curr_date, curr_open, curr_high, curr_low, curr_close, curr_volume,
                "holding", entry_price, entry_take_profit, entry_stop_loss, "", 0, 0, f"bars_held={bars_held}"
            )

        # If not in position, keep equity flat (append last_equity)
        if position == 0:
            equity.append(last_equity)

    # If still in position at the end, close at last close
    if position == 1:
        pnl = (close_prices[-1] - entry_price) / entry_price
        pnl = -pnl if REVERT_PROFIT else pnl
        last_equity = last_equity * (1 + pnl)
        equity.append(last_equity)
        log_trade(entry_date, entry_price, dates[-1], close_prices[-1], pnl, "final_close")
        pnl_detail = f"({close_prices[-1]:.6f} - {entry_price:.6f}) / {entry_price:.6f}"
        log_detailed(
            dates[-1], open_prices[-1], high_prices[-1], low_prices[-1], close_prices[-1], volumes[-1],
            "exit", entry_price, entry_take_profit, entry_stop_loss, "final_close", pnl, last_equity - (last_equity / (1 + pnl)), pnl_detail
        )
    else:
        equity.append(last_equity)

    # Write detailed log to CSV
    if WRITE_CSV:
        log_filename = f"data/lstm_backtest_log_{timestamp}.csv"
        log_columns = [
            'datetime', 'open', 'high', 'low', 'close', 'volume',
            'state', 'entry_price', 'take_profit', 'stop_loss', 'exit_method', 'pnl', 'abs_pnl', 'pnl_detail'
        ]
        with open(log_filename, 'w', newline='') as csvfile:
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
    # print(f"Backtest Results (Long-only, threshold={threshold*100:.2f}%):")
    # print(f"  Total Return: {total_return*100:.2f}%")
    # print(f"  Number of Trades: {number_of_trades}")
    # print(f"  Win Rate: {win_rate*100:.2f}%")
    # print(f"  Max Drawdown: {max_drawdown*100:.2f}%")

    # Plot equity curve with price overlay (separate Y-axes)
    plt.figure(figsize=(12, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    min_len = min(len(dates), len(equity), len(close_prices))
    plot_dates = dates[:min_len]
    plot_equity = equity[:min_len]

    l1, = ax1.plot(plot_dates, plot_equity, label='Equity Curve', color='blue')
    ax1.set_ylabel('Equity ($)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    l2, = ax2.plot(plot_dates, close_prices[:min_len], label='Price', color='orange', alpha=0.5)
    ax2.set_ylabel('Price', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.title(f'Equity Curve {symbol} with Price Overlay')
    ax1.set_xlabel('Date')
    ax1.grid(True)

    lines = [l1, l2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.savefig(f"data/lstm_equity_curve_{symbol}_{timestamp}.png")

    return trade_log, equity, total_return, number_of_trades, win_rate, max_drawdown
