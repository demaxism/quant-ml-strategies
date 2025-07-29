import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

# ---------------------------
# Technical Indicators
# ---------------------------
def add_technical_indicators(df):
    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['BB_Upper'] = df['MA20'] + (df['close'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['MA20'] - (df['close'].rolling(window=20).std() * 2)

    # Exponential Moving Averages
    df['EMA10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

    # Moving Averages
    df['SMA10'] = df['close'].rolling(window=10).mean()
    df['SMA50'] = df['close'].rolling(window=50).mean()

    # Momentum
    df['Momentum'] = df['close'] - df['close'].shift(10)

    # Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

    # Drop NaN values resulting from indicator calculations
    df = df.dropna().reset_index(drop=True)
    return df

# ---------------------------
# Data Loading & Preprocessing
# ---------------------------
def load_data(filepath, n_hours=24):
    # Load feather file
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_feather(filepath)
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    # Print time range and preview
    # 输出时间范围
    start_time = df['date'].min()
    end_time = df['date'].max()
    print(f"\n📅 Time range in dataset: {start_time} → {end_time}")
    print("Data preview (first 5 rows):")
    print(df.head(5))
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    df = df.dropna(subset=required_cols)
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Normalize price, volume, and technical indicators
    scaler = MinMaxScaler()
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'MACD_Signal',
                   'BB_Upper', 'BB_Lower', 'EMA10', 'EMA50', 'SMA10', 'SMA50', 'Momentum', 'VWAP']
    features = df[feature_cols].values
    features_scaled = scaler.fit_transform(features)
    
    # Generate sliding window samples and labels
    X, y = [], []
    for i in range(len(df) - n_hours - 1):
        window = features_scaled[i:i+n_hours]
        # Label: next hour close >1% than current close
        close_now = df['close'].iloc[i+n_hours-1]
        close_next = df['close'].iloc[i+n_hours]
        label = 1 if (close_next - close_now) / close_now > 0.005 else 0  # Relaxed to >0.5%
        X.append(window)
        y.append(label)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y, scaler

# ---------------------------
# Data Split
# ---------------------------
def split_data(X, y, train_ratio=0.7, val_ratio=0.2):
    n_total = len(X)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ---------------------------
# PyTorch Dataset
# ---------------------------
class CryptoDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------
# LSTM Model
# ---------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out.squeeze(-1)

# ---------------------------
# Training Function
# ---------------------------
def train_model(model, loaders, device, epochs=10, lr=1e-3):
    train_loader, val_loader = loaders

    # Compute pos_weight for BCELoss
    # Get all y in train_loader
    all_y = []
    for _, y_batch in train_loader:
        all_y.extend(y_batch.numpy())
    all_y = np.array(all_y)
    n_pos = np.sum(all_y == 1)
    n_neg = np.sum(all_y == 0)
    print(f"Train set label=1 count: {n_pos} / {len(all_y)} ({n_pos/len(all_y):.4%})")
    if n_pos == 0:
        pos_weight = torch.tensor(1.0, device=device)
        print("Warning: No positive samples in training set. pos_weight set to 1.0")
    else:
        pos_weight = torch.tensor(n_neg / n_pos, device=device)
        print(f"Using pos_weight={pos_weight.item():.2f} for BCELoss")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        avg_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        # Validation
        acc, prec, rec, f1 = evaluate_model(model, val_loader, device, verbose=False)
        val_metrics['acc'].append(acc)
        val_metrics['prec'].append(prec)
        val_metrics['rec'].append(rec)
        val_metrics['f1'].append(f1)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | Val Prec: {prec:.4f} | Val Rec: {rec:.4f} | Val F1: {f1:.4f}")
    return train_losses, val_metrics

# ---------------------------
# Evaluation Function
# ---------------------------
def evaluate_model(model, loader, device, verbose=True):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).cpu().numpy()
            probs = 1 / (1 + np.exp(-outputs))  # sigmoid
            preds = (probs > 0.5).astype(int)
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    if verbose:
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    return acc, prec, rec, f1

# ---------------------------
# Backtest Function
# ---------------------------
def backtest(model, loader, device, threshold=0.5):
    model.eval()
    n_signals = 0
    n_wins = 0
    y_true, y_prob = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))  # sigmoid
            y_true.extend(y_batch.numpy())
            y_prob.extend(probs)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    signals = y_prob > threshold
    n_signals = np.sum(signals)
    n_wins = np.sum((signals) & (y_true == 1))
    winrate = n_wins / n_signals if n_signals > 0 else 0.0
    # Diagnostic prints
    print(f"Test set label positive count: {np.sum(y_true==1)} / {len(y_true)}")
    print(f"Predicted probability stats: min={y_prob.min():.4f}, max={y_prob.max():.4f}, mean={y_prob.mean():.4f}")
    print(f"Backtest: Buy signals: {n_signals}, Wins: {n_wins}, Winrate: {winrate:.4f}")
    return n_signals, n_wins, winrate, y_prob

# ---------------------------
# Plotting Function (Optional)
# ---------------------------
def plot_metrics(train_losses, val_metrics):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(val_metrics['acc'], label='Val Acc')
    plt.plot(val_metrics['f1'], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------
# Main Function
# ---------------------------
def main():
    # Config
    data_path = "data/BTC_USDT-1h.feather"
    n_hours = 24
    batch_size = 32
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, scaler = load_data(data_path, n_hours=n_hours)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 2. Build datasets and loaders
    train_dataset = CryptoDataset(X_train, y_train)
    val_dataset = CryptoDataset(X_val, y_val)
    test_dataset = CryptoDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Build model
    model = LSTMClassifier(input_size=16, hidden_size=64, num_layers=2).to(device)

    # Print total and per-layer parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()}")

    # 4. Train model
    print("Training model...")
    train_losses, val_metrics = train_model(model, (train_loader, val_loader), device, epochs=epochs)

    # 5. Evaluate on validation set
    print("Validation set performance:")
    evaluate_model(model, val_loader, device)

    # 6. Backtest on test set
    print("Backtest on test set:")
    n_signals, n_wins, winrate, y_prob = backtest(model, test_loader, device, threshold=0.5)

    # 7. Plot metrics
    plot_metrics(train_losses, val_metrics)

    # 8. Print summary
    print(f"Test set buy signals: {n_signals}, Wins: {n_wins}, Winrate: {winrate:.4f}")

if __name__ == "__main__":
    main()
