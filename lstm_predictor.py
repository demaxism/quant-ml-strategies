# explain: This script implements a generic LSTM-based price predictor for financial time series data.
# It loads data from a feather file, normalizes it, creates sequences for LSTM input,
# trains an LSTM model, and evaluates its predictions. It also includes a long-only trading
# strategy backtest based on the predicted high and low prices.
# The script can be run from the command line with options for the data file and sequence length
# sample: python lstm_predictor.py --datafile data/ETH_USDT-4h.feather --seq_len 48


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import re
import os

def parse_symbol_timeframe(filepath):
    # Try to extract SYMBOL-TIMEFRAME from filename, e.g. ETH_USDT-4h
    basename = os.path.basename(filepath)
    match = re.search(r'([A-Z]+_[A-Z]+)-([0-9]+[a-zA-Z]+)', basename)
    if match:
        symbol, timeframe = match.group(1), match.group(2)
        return symbol, timeframe
    else:
        return "Unknown", "Unknown"

def main():
    parser = argparse.ArgumentParser(description="Generic LSTM price predictor")
    parser.add_argument('--datafile', type=str, default='data/ETH_USDT-1h.feather',
                        help='Path to input feather file (e.g., data/ETH_USDT-4h.feather)')
    parser.add_argument('--seq_len', type=int, default=48,
                        help='Number of past candles to use for prediction')
    args = parser.parse_args()

    datafile = args.datafile
    SEQ_LEN = args.seq_len

    if not os.path.exists(datafile):
        raise FileNotFoundError(f"Data file not found: {datafile}")

    symbol, timeframe = parse_symbol_timeframe(datafile)

    # Load data
    df = pd.read_feather(datafile)
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df['change'] = df['close'].pct_change().fillna(0)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    print('start analyze')
    print(df.head())

    # Normalize
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    # Sequence
    X, y = [], []
    for i in range(len(scaled) - SEQ_LEN - 1):
        X.append(scaled[i:i+SEQ_LEN])
        y.append(scaled[i+SEQ_LEN][1:3])  # predict [high, low]
    X, y = np.array(X), np.array(y)

    # Train/test split
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Print time ranges for training and test (backtest) sets
    train_start = df.index[SEQ_LEN]
    train_end = df.index[SEQ_LEN + split - 1]
    test_start = df.index[SEQ_LEN + split + 1]
    test_end = df.index[SEQ_LEN + split + len(X_test)]
    print(f"Training set time range: {train_start} to {train_end}")
    print(f"Backtest (test) set time range: {test_start} to {test_end}")

    # Dataset
    class PriceDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx): return self.X[idx], self.y[idx]

    train_loader = DataLoader(PriceDataset(X_train, y_train), batch_size=64, shuffle=True)

    # Model
    class LSTMPriceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(6, 64, 2, batch_first=True)
            self.fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(h[-1])

    model = LSTMPriceModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Train
    for epoch in range(10):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    # Predict
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    # Inverse transform
    def inverse_high_low(vals):
        dummy = np.zeros((len(vals), 6))
        dummy[:, 1:3] = vals
        return scaler.inverse_transform(dummy)[:, 1:3]

    true = inverse_high_low(y_test)
    predicted = inverse_high_low(pred)

    # Step 1: Get full index of original data
    date_index = df.index[SEQ_LEN + split + 1:]  # +1 for prediction shift

    # Step 2: Select the first 100 timestamps for the plotted range
    plot_dates = date_index[:100]

    # Print chart start date
    if len(plot_dates) > 0:
        print("Chart start date:", plot_dates[0].strftime('%Y-%m-%d'))

    # Step 3: Plot with datetime x-axis
    plt.figure(figsize=(14, 6))
    plt.plot(plot_dates, true[:100, 0], label="True High")
    plt.plot(plot_dates, predicted[:100, 0], label="Pred High", linestyle="--")
    plt.plot(plot_dates, true[:100, 1], label="True Low")
    plt.plot(plot_dates, predicted[:100, 1], label="Pred Low", linestyle="--")

    plt.legend()
    plt.title(f"{symbol} LSTM Predicted vs True High/Low ({timeframe} timeframe)")
    plt.xlabel("Datetime")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"lstm_predictions_{symbol}_{timeframe}.png")

    # === Long-only Trading Strategy Backtest ===
    def backtest_long_only_strategy(true, predicted, date_index, threshold=0.008, allowance=0.002):
        """
        true: [N, 2] array of true [high, low] (not used here), but we use close prices from df
        predicted: [N, 2] array of predicted [high, low]
        date_index: DatetimeIndex for test set
        threshold: float, e.g. 0.002 for 0.2%
        allowance: float, take profit/stop loss trigger as a fraction (default 0.002 = 0.2%)
        
        Exit logic:
        - If in position, check in order:
            1. Stop loss: if next bar's low <= pred_low * (1 + allowance), exit at pred_low * (1 + allowance)
            2. Take profit: if next bar's high >= pred_high * (1 - allowance), exit at pred_high * (1 - allowance)
            3. Otherwise, exit at next bar's close
        - Only one exit per trade (whichever is triggered first)
        - Only one entry per bar, and no immediate re-entry after exit
        - Equity is only updated when a trade is closed or at the end of the bar if flat
        """
        # For entry/exit, we need the true close and high prices for the test set
        close_prices = df['close'].values[SEQ_LEN + split + 1:]
        close_prices = close_prices[:len(predicted)]
        high_prices = df['high'].values[SEQ_LEN + split + 1:]
        high_prices = high_prices[:len(predicted)]
        dates = date_index[:len(predicted)]

        equity = [1.0]  # start with $1
        position = 0    # 0 = flat, 1 = long
        entry_price = 0
        entry_date = None
        trade_log = []
        last_equity = equity[-1]
        for i in range(len(predicted) - 1):  # last prediction can't be traded (no next close)
            curr_close = close_prices[i]
            next_close = close_prices[i+1]
            next_high = high_prices[i+1]
            next_low = df['low'].values[SEQ_LEN + split + 1 + i + 1]
            pred_high = predicted[i, 0]
            pred_low = predicted[i, 1]
            exited_this_bar = False

            # Exit logic: only if in position
            if position == 1:
                stop_loss_price = pred_low * (1 + allowance)
                take_profit_price = pred_high * (1 - allowance)
                # 1. Stop loss
                if next_low <= stop_loss_price:
                    pnl = (stop_loss_price - entry_price) / entry_price
                    last_equity = last_equity * (1 + pnl)
                    equity.append(last_equity)
                    trade_log.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': dates[i+1],
                        'exit_price': stop_loss_price,
                        'pnl': pnl,
                        'exit_type': 'stop_loss'
                    })
                    position = 0
                    exited_this_bar = True
                # 2. Take profit (only if stop loss not triggered)
                elif next_high >= take_profit_price:
                    pnl = (take_profit_price - entry_price) / entry_price
                    last_equity = last_equity * (1 + pnl)
                    equity.append(last_equity)
                    trade_log.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': dates[i+1],
                        'exit_price': take_profit_price,
                        'pnl': pnl,
                        'exit_type': 'take_profit'
                    })
                    position = 0
                    exited_this_bar = True
                # 3. Exit at next close (only if neither triggered)
                else:
                    pnl = (next_close - entry_price) / entry_price
                    last_equity = last_equity * (1 + pnl)
                    equity.append(last_equity)
                    trade_log.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': dates[i+1],
                        'exit_price': next_close,
                        'pnl': pnl,
                        'exit_type': 'close'
                    })
                    position = 0
                    exited_this_bar = True

            # Entry logic: only if not in position and did not just exit
            if position == 0 and not exited_this_bar and pred_high > curr_close * (1 + threshold):
                position = 1
                entry_price = curr_close
                entry_date = dates[i]

            # If not in position, keep equity flat (append last_equity)
            if position == 0:
                equity.append(last_equity)

        # If still in position at the end, close at last close
        if position == 1:
            pnl = (close_prices[-1] - entry_price) / entry_price
            last_equity = last_equity * (1 + pnl)
            equity.append(last_equity)
            trade_log.append({
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': dates[-1],
                'exit_price': close_prices[-1],
                'pnl': pnl,
                'exit_type': 'final_close'
            })
        else:
            equity.append(last_equity)

        # Compute stats
        returns = np.array([t['pnl'] for t in trade_log])
        total_return = equity[-1] - 1.0
        win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
        max_drawdown = np.max(np.maximum.accumulate(equity[:-1]) - equity[:-1])
        print(f"Backtest Results (Long-only, threshold={threshold*100:.2f}%):")
        print(f"  Total Return: {total_return*100:.2f}%")
        print(f"  Number of Trades: {len(trade_log)}")
        print(f"  Win Rate: {win_rate*100:.2f}%")
        print(f"  Max Drawdown: {max_drawdown*100:.2f}%")

        # Plot equity curve with price overlay (separate Y-axes)
        plt.figure(figsize=(12, 5))
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        # Plot equity curve (right Y-axis)
        l1, = ax1.plot(dates, equity[:-1], label='Equity Curve', color='blue')
        ax1.set_ylabel('Equity ($)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Plot price curve (left Y-axis, semi-transparent orange)
        l2, = ax2.plot(dates, close_prices[:len(dates)], label='Price', color='orange', alpha=0.5)
        ax2.set_ylabel('Price', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        # Title, grid, legend, etc.
        plt.title('Equity Curve (Long-only Strategy) with Price Overlay')
        ax1.set_xlabel('Date')
        ax1.grid(True)

        # Combine legends
        lines = [l1, l2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.tight_layout()
        plt.show()

        return trade_log, equity

    # Run backtest with default threshold, or allow override via env var
    threshold = float(os.environ.get('LSTM_STRATEGY_THRESHOLD', 0.002))
    backtest_long_only_strategy(true, predicted, date_index, threshold)

if __name__ == "__main__":
    main()
