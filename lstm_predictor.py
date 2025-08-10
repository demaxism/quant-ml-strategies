# explain: This script implements a generic LSTM-based price predictor for financial time series data.
# It loads data from a feather file, normalizes it, creates sequences for LSTM input,
# trains an LSTM model, and evaluates its predictions. It also includes a long-only trading
# strategy backtest based on the predicted high and low prices.
#   For each bar in the test set, it predicts the high and low prices PREDICT_AHEAD bars ahead.
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
from backtest_simulate import backtest_realtime_lstm

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
    parser.add_argument('--bt_from', type=str, default=None,
                        help='Backtest start date (e.g., 2025-03-01)')
    parser.add_argument('--bt_until', type=str, default=None,
                        help='Backtest end date (e.g., 2025-03-01)')
    parser.add_argument('--seq_len', type=int, default=12,
                        help='Number of past candles to use for prediction')
    parser.add_argument('--predict_ahead', type=int, default=2,
                        help='Number of bars ahead to predict (N bar after the sequence)')
    parser.add_argument('--model_file', type=str, default=None,
                        help='Path to save/load the LSTM model weights')
    parser.add_argument('--n_hold', type=int, default=5,
                        help='Number of bars to hold after entry (default: 5)')
    parser.add_argument('--n_turn', type=int, default=1,
                        help='Number of times to repeat training and backtesting (default: 1)')
    parser.add_argument('--revert_profit', action='store_true',
                        help='If set, all profit becomes loss and all loss becomes profit (simulate shorting)')
    parser.add_argument('--csv', action='store_true',
                        help='If set, enable CSV/Figure logging')
    parser.add_argument('--fine_timeframe', type=str, default='1h',
                        help='Timeframe for the fine data (e.g., 1h, 30m)')
    args = parser.parse_args()

    datafile = args.datafile
    fine_datafile = datafile.replace('-4h', f'-{args.fine_timeframe}')  # Fine data file path
    SEQ_LEN = args.seq_len
    PREDICT_AHEAD = args.predict_ahead
    N_HOLD = args.n_hold
    model_file = args.model_file
    no_train = model_file is not None
    N_TURN = args.n_turn
    REVERT_PROFIT = args.revert_profit
    WRITE_CSV = args.csv  # Enable CSV logging if --csv is set
    # choose to plot first or last N predictions
    plot_first = None  # Default to plot first 100 predictions
    plot_last = 500  # Default to plot last 100 predictions
    LAYERS = 2  # Number of LSTM layers
    BT_FROM = args.bt_from
    BT_UNTIL = args.bt_until
    threshold = 0.016 # Default threshold for entry logic

    if not os.path.exists(datafile):
        raise FileNotFoundError(f"Data file not found: {datafile}")

    symbol, timeframe = parse_symbol_timeframe(datafile)

    # Load data
    df = pd.read_feather(datafile)
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df['change'] = df['close'].pct_change().fillna(0)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Load fine data if available
    fine_df = None
    if os.path.exists(fine_datafile):
        fine_df = pd.read_feather(fine_datafile)
        fine_df = fine_df[['date', 'open', 'high', 'low', 'close', 'volume']]
        fine_df['change'] = fine_df['close'].pct_change().fillna(0)
        fine_df['date'] = pd.to_datetime(fine_df['date'])
        fine_df.set_index('date', inplace=True)

    # print('start analyze')
    # print("==== Data Overview ====")
    # print(df.head())
    # print("==== Fine Data Overview ====")
    # if fine_df is not None:
    #     print(fine_df.head())
    # else:
    #     print("No fine data available.")

    # print current timestamp
    print(f"Current timestamp: {timestamp}")

    # print current git repo branch and commit
    try:
        import subprocess
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        print(f"Current Git branch: {branch}")
        print(f"Current Git commit: {commit}")
    except Exception as e:
        print(f"Error getting Git info: {e}")
        branch, commit = "unknown", "unknown"

    # Normalize
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    # Sequence
    # For each i, X = sequence of SEQ_LEN (scaled), y = [high_return, low_return] of (SEQ_LEN + PREDICT_AHEAD - 1)-th bar after i (computed from original prices)
    X, y = [], []
    df_values = df.values  # original (unscaled) values, columns: ['open', 'high', 'low', 'close', 'volume', 'change']
    for i in range(len(scaled) - SEQ_LEN - PREDICT_AHEAD + 1):
        X.append(scaled[i:i+SEQ_LEN])
        # Use original prices for return calculation
        current_close = df_values[i+SEQ_LEN-1][3]  # close column
        future_high = df_values[i+SEQ_LEN+PREDICT_AHEAD-1][1]  # high column
        future_low = df_values[i+SEQ_LEN+PREDICT_AHEAD-1][2]   # low column
        
        high_return = (future_high - current_close) / current_close
        low_return = (future_low - current_close) / current_close
        y.append([high_return, low_return])
    X, y = np.array(X), np.array(y)

    # Train/test split
    training_percentage = 0.7
    split = int(len(X) * training_percentage)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Print time ranges for training and test (backtest) sets
    train_start = df.index[SEQ_LEN]
    train_end = df.index[SEQ_LEN + split - 1]
    test_start = df.index[SEQ_LEN + split + 1]
    test_end = df.index[SEQ_LEN + split + len(X_test) - 1]
    print(f"Training set time range: {train_start} to {train_end}")
    print(f"Backtest (test) set time range: {test_start} to {test_end}")
    print(f"manual backtest start: {BT_FROM}, end: {BT_UNTIL}")

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
            self.lstm = nn.LSTM(6, 64, LAYERS, batch_first=True)
            self.fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(h[-1])

    def run_one_turn(turn_idx, timestamp, plot_first=None, plot_last=None):
        model = LSTMPriceModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        this_model_file = model_file
        this_timestamp = timestamp
        if not no_train:
            # Train with early stopping
            best_loss = float('inf')
            patience = 3  # Number of epochs to wait for improvement
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
                print(f"Turn {turn_idx+1} Epoch {epoch+1} Loss: {avg_loss:.4f}")

                # Early stopping logic
                if best_loss - avg_loss > min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1} (no significant improvement for {patience} epochs).")
                        break

            this_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_turn{turn_idx+1}"
            this_model_file = f"data/log/model_{symbol}_Ly{LAYERS}_Sq{SEQ_LEN}_Ah{PREDICT_AHEAD}_{this_timestamp}.pt"
            torch.save(model.state_dict(), this_model_file)
            print(f"Saved trained model weights to {this_model_file}.")
        else:
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Model file not found: {model_file}")
            model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
            print(f"Loaded model weights from {model_file}. Skipping training.")

        # Predict
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

        # Reconstruct high/low prices from predicted/true returns and current close price
        def reconstruct_high_low_from_return(returns, closes):
            # returns: shape (N, 2), closes: shape (N,)
            # high = close * (1 + high_return), low = close * (1 + low_return)
            high = closes * (1 + returns[:, 0])
            low = closes * (1 + returns[:, 1])
            return np.stack([high, low], axis=1)

        # Get the close price at the end of each input sequence in the test set (for plotting)
        test_close = []
        for i in range(split, split + len(X_test)):
            test_close.append(df_values[i+SEQ_LEN-1][3])  # original close price
        test_close = np.array(test_close)

        true = reconstruct_high_low_from_return(y_test, test_close)
        predicted = reconstruct_high_low_from_return(pred, test_close)

        # === Deviation Metric: Mean Absolute Error (MAE) for High and Low ===
        high_mae = np.mean(np.abs(predicted[:, 0] - true[:, 0]))
        low_mae = np.mean(np.abs(predicted[:, 1] - true[:, 1]))
        print(f"Prediction Deviation Metrics: High MAE: {high_mae:.4f}, Low MAE: {low_mae:.4f} ")

        # Step 1: Get full index of original data
        date_index = df.index[SEQ_LEN + split:]  # fixed: removed +1 to match X_test length

        if WRITE_CSV:
            # Step 2: Select the plotting range based on user input
            if plot_first is not None and plot_last is not None:
                raise ValueError("Only one of --plot_first or --plot_last can be set.")
            if plot_first is not None:
                plot_dates = date_index[:plot_first]
                true_plot = true[:plot_first]
                predicted_plot = predicted[:plot_first]
            elif plot_last is not None:
                plot_dates = date_index[-plot_last:]
                true_plot = true[-plot_last:]
                predicted_plot = predicted[-plot_last:]
            else:
                plot_dates = date_index[:100]
                true_plot = true[:100]
                predicted_plot = predicted[:100]

            # Step 3: Plot with datetime x-axis
            plt.figure(figsize=(14, 6))
            plt.plot(plot_dates, true_plot[:, 0], label="True High (reconstructed)")
            plt.plot(plot_dates, predicted_plot[:, 0], label="Pred High (reconstructed)", linestyle="--")
            plt.plot(plot_dates, true_plot[:, 1], label="True Low (reconstructed)")
            plt.plot(plot_dates, predicted_plot[:, 1], label="Pred Low (reconstructed)", linestyle="--")

            plt.legend()
            plt.title(f"{symbol} LSTM Predicted vs True High/Low (returns target, {timeframe} timeframe)")
            plt.xlabel("Datetime")
            plt.ylabel("Price")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"data/log/lstm_predictions_{symbol}_{timeframe}_turn{turn_idx+1}.png")

        # === Long-only Trading Strategy Backtest ===
        
        if True:
            # # # Use bar-by-bar real-time backtest
            trade_log, equity, total_return, number_of_trades, win_rate, max_drawdown = backtest_realtime_lstm(
                model, df, split, SEQ_LEN, PREDICT_AHEAD, N_HOLD, this_timestamp, scaler, WRITE_CSV, REVERT_PROFIT, threshold, symbol=symbol, fine_df=fine_df, BT_FROM=BT_FROM, BT_UNTIL=BT_UNTIL, test_start=test_start
            )
        else:
            trade_log, equity, total_return, number_of_trades, win_rate, max_drawdown = backtest_long_only_strategy(
                true, predicted, date_index, df, split, SEQ_LEN, N_HOLD, this_timestamp, WRITE_CSV, REVERT_PROFIT, threshold, symbol=symbol
            )
        print(f"Backtest Metrics (Turn {turn_idx+1}):")
        print(f"  Total Return: {total_return*100:.2f}%")
        print(f"  Number of Trades: {number_of_trades}")
        print(f"  Win Rate: {win_rate*100:.2f}%")
        print(f"  Max Drawdown: {max_drawdown*100:.2f}%")
        return this_model_file, total_return, number_of_trades, win_rate, max_drawdown, high_mae, low_mae

    run_turns = N_TURN if not no_train else 1
    results = []
    for turn in range(run_turns):
        model_file_name, total_return, number_of_trades, win_rate, max_drawdown, high_mae, low_mae = run_one_turn(turn, timestamp, plot_first, plot_last)
        results.append({
            'model_file': model_file_name,
            'total_return': total_return,
            'number_of_trades': number_of_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'high_mae': high_mae,
            'low_mae': low_mae
        })

        # Print top 3 results so far, sorted by total_return / max_drawdown
        def _sort_key(row):
            return row['total_return'] / row['max_drawdown'] if row['max_drawdown'] != 0 else float('inf')
        top3 = sorted(results, key=_sort_key, reverse=True)[:3]
        print("\nTop 3 results so far (by total_return / max_drawdown):")
        print("{:<35} {:>12} {:>15}".format("Model File", "Total Return", "Max Drawdown"))
        for row in top3:
            print("{:<35} {:>12.2f}% {:>15.2f}%".format(
                (os.path.basename(row['model_file']))[-20:],
                row['total_return']*100,
                row['max_drawdown']*100
            ))

    # Sort results by (total_return / max_drawdown) descending, using column names
    def sort_key(row):
        return row['total_return'] / row['max_drawdown'] if row['max_drawdown'] != 0 else float('inf')
    results_sorted = sorted(results, key=sort_key, reverse=True)

    # Print summary table
    print("\n=== Summary of All Turns ===")
    print("{:<35} {:>12} {:>15} {:>10} {:>15} {:>10} {:>10}".format("Model File", "Total Return", "Num Trades", "Win Rate", "Max Drawdown", "High MAE", "Low MAE"))
    for row in results_sorted:
        print("{:<35} {:>12.2f}% {:>15} {:>10.2f}% {:>15.2f}% {:>10.2f} {:>10.2f} ".format(
            (os.path.basename(row['model_file']))[-20:],
            row['total_return']*100,
            row['number_of_trades'],
            row['win_rate']*100,
            row['max_drawdown']*100,
            row['high_mae'],
            row['low_mae']
        ))
    return results_sorted

if __name__ == "__main__":
    main()
