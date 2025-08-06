# explain: This script implements a generic LSTM-based price predictor for financial time series data.
# It loads data from a feather file, normalizes it, creates sequences for LSTM input,
# trains an LSTM model, and evaluates its predictions. It also includes a long-only trading
# strategy backtest based on the predicted high and low prices.
#   For each bar in the test set, it predicts the high and low prices N_HOLD bars ahead.
#   It entry logic is based on a threshold above the current close price,
#   and exit logic includes stop loss, take profit, and maximum hold time (N_HOLD).
# The script can be run from the command line with options for the data file and sequence length
# sample: python lstm_predictor.py --datafile data/ETH_USDT-4h.feather --seq_len 48
# Load model sameple: python lstm_predictor.py --datafile data/ETH_USDT-4h.feather --model_file 'lstm_model.pth' --no_train


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
from datetime import datetime
from backtest_long_only_strategy import backtest_long_only_strategy

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

WRITE_CSV = False  # Set to False to disable CSV logging

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
    parser.add_argument('--model_file', type=str, default='lstm_model.pth',
                        help='Path to save/load the LSTM model weights')
    parser.add_argument('--no_train', action='store_true',
                        help='If set, load model from file and skip training')
    parser.add_argument('--n_hold', type=int, default=1,
                        help='Number of bars ahead to predict (N_HOLD-th bar after the sequence)')
    args = parser.parse_args()

    datafile = args.datafile
    SEQ_LEN = args.seq_len
    N_HOLD = args.n_hold
    model_file = args.model_file
    no_train = args.no_train

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
    # For each i, X = sequence of SEQ_LEN, y = [high, low] of (SEQ_LEN + N_HOLD - 1)-th bar after i
    X, y = [], []
    for i in range(len(scaled) - SEQ_LEN - N_HOLD + 1):
        X.append(scaled[i:i+SEQ_LEN])
        y.append(scaled[i+SEQ_LEN+N_HOLD-1][1:3])  # predict [high, low] of N_HOLD-th bar after the sequence
    X, y = np.array(X), np.array(y)

    # Train/test split
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Print time ranges for training and test (backtest) sets
    train_start = df.index[SEQ_LEN]
    train_end = df.index[SEQ_LEN + split - 1]
    test_start = df.index[SEQ_LEN + split + 1]
    test_end = df.index[SEQ_LEN + split + len(X_test) - 1]
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

    if no_train:
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        print(f"Loaded model weights from {model_file}. Skipping training.")
    else:
        # Train with early stopping
        best_loss = float('inf')
        patience = 2  # Number of epochs to wait for improvement
        patience_counter = 0
        min_delta = 1e-4  # Minimum improvement to reset patience

        for epoch in range(10):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

            # Early stopping logic
            if best_loss - avg_loss > min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} (no significant improvement for {patience} epochs).")
                    break

        model_file = f"data/model_{timestamp}.pt"
        torch.save(model.state_dict(), model_file)
        print(f"Saved trained model weights to {model_file}.")

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
    plt.savefig(f"data/lstm_predictions_{symbol}_{timeframe}.png")

    # === Long-only Trading Strategy Backtest ===
    threshold = float(os.environ.get('LSTM_STRATEGY_THRESHOLD', 0.002))
    trade_log, equity, total_return, number_of_trades, win_rate, max_drawdown = backtest_long_only_strategy(
        true, predicted, date_index, df, split, SEQ_LEN, timestamp, WRITE_CSV, threshold
    )
    # Print or return the metrics so the caller can capture them
    print(f"Backtest Metrics:")
    print(f"  Total Return: {total_return*100:.2f}%")
    print(f"  Number of Trades: {number_of_trades}")
    print(f"  Win Rate: {win_rate*100:.2f}%")
    print(f"  Max Drawdown: {max_drawdown*100:.2f}%")
    return total_return, number_of_trades, win_rate, max_drawdown

if __name__ == "__main__":
    main()
