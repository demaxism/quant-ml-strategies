import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os

# ---------------------------
# 超参数配置（便于统一调整）
# ---------------------------
DATA_PATH = "data/BTC_USDT-4h.feather"  # 数据文件路径
N_HOURS = 84        # 输入窗口长度（蜡烛的个数）
BATCH_SIZE = 32     # 批大小，一般64或32，更大会快但内存消耗大 太大可能影响模型收敛效果
EPOCHS = 50         # 训练轮数
LEARNING_RATE = 1e-3  # 学习率

# LSTM模型结构参数
LSTM_INPUT_SIZE = 16    # 输入特征数（一般不用改）
LSTM_HIDDEN_SIZE = 128  # LSTM隐藏单元数
LSTM_NUM_LAYERS = 3     # LSTM层数
LSTM_DROPOUT = 0.3      # Dropout比例
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------------------------
# Early Stopping
# ---------------------------
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_f1 = None
        self.early_stop = False

    def __call__(self, f1):
        if self.best_f1 is None:
            self.best_f1 = f1
        elif f1 < self.best_f1:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_f1 = f1
            self.counter = 0

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
def load_data(filepath, n_hours=84):
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
        # Label: next hour close >0.25% than current close
        close_now = df['close'].iloc[i+n_hours-1]
        close_next = df['close'].iloc[i+n_hours] # 预测下一根蜡烛线
        label = 1 if (close_next - close_now) / close_now > 0.01 else 0  # Predicts >1% increase
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
# Data Balancing
# ---------------------------
def balance_data(X_train, y_train):
    # Reshape X_train for SMOTE: (samples, features)
    n_samples, n_hours, n_features = X_train.shape
    X_train_flat = X_train.reshape(n_samples, -1)
    
    smote = SMOTE()
    X_train_res, y_train_res = smote.fit_resample(X_train_flat, y_train)
    
    # Reshape back to (samples, n_hours, n_features)
    X_train_res = X_train_res.reshape(-1, n_hours, n_features)
    y_train_res = y_train_res
    print(f"After SMOTE, Train: {len(X_train_res)} samples")
    return X_train_res, y_train_res

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
    def __init__(self, input_size=LSTM_INPUT_SIZE, hidden_size=LSTM_HIDDEN_SIZE, num_layers=LSTM_NUM_LAYERS, dropout=LSTM_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 256, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(256 * 2, 1)
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    train_losses = []
    val_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
    early_stopping = EarlyStopping(patience=5, verbose=True)
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
        acc, prec, rec, f1, y_val_true, y_val_prob = evaluate_model(model, val_loader, device, verbose=False)
        val_metrics['acc'].append(acc)
        val_metrics['prec'].append(prec)
        val_metrics['rec'].append(rec)
        val_metrics['f1'].append(f1)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | Val Prec: {prec:.4f} | Val Rec: {rec:.4f} | Val F1: {f1:.4f}")
        # Scheduler step
        scheduler.step(f1)
        # Early Stopping
        early_stopping(f1)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break
    return train_losses, val_metrics, y_val_true, y_val_prob

# ---------------------------
# Evaluation Function
# ---------------------------
def evaluate_model(model, loader, device, verbose=True):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).cpu().numpy()
            probs = 1 / (1 + np.exp(-outputs))  # sigmoid
            preds = (probs > 0.5).astype(int)
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    if verbose:
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    return acc, prec, rec, f1, np.array(y_true), np.array(y_prob)

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
# Find Optimal Threshold Function
# ---------------------------
def find_optimal_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    max_f1 = f1_scores[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.4f} with F1 Score: {max_f1:.4f}")
    return optimal_threshold

# ---------------------------
# Main Function
# ---------------------------
def main():
    # Config
    data_path = DATA_PATH
    n_hours = N_HOURS
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    lr = LEARNING_RATE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, scaler = load_data(data_path, n_hours=n_hours)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 2. Balance training data
    print("Balancing training data with SMOTE...")
    X_train_res, y_train_res = balance_data(X_train, y_train)
    print(f"After SMOTE, Train: {len(X_train_res)} samples")

    # 3. Build datasets and loaders
    train_dataset = CryptoDataset(X_train_res, y_train_res)
    val_dataset = CryptoDataset(X_val, y_val)
    test_dataset = CryptoDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. Build model
    model = LSTMClassifier(
        input_size=LSTM_INPUT_SIZE,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT
    ).to(device)

    # Print total and per-layer parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()}")

    # 5. Train model
    print("Training model...")
    train_losses, val_metrics, y_val_true, y_val_prob = train_model(model, (train_loader, val_loader), device, epochs=epochs, lr=lr)

    # 6. Find optimal threshold based on validation set
    print("Finding optimal threshold based on validation set...")
    optimal_threshold = find_optimal_threshold(y_val_true, y_val_prob)

    # 7. Evaluate on validation set with optimal threshold
    print("Validation set performance with optimal threshold:")
    n_signals_val, n_wins_val, winrate_val, _ = backtest(model, val_loader, device, threshold=optimal_threshold)

    # 8. Backtest on test set with optimal threshold
    print("Backtest on test set with optimal threshold:")
    n_signals, n_wins, winrate, y_prob_test = backtest(model, test_loader, device, threshold=0.25)

    # 9. Plot metrics
    plot_metrics(train_losses, val_metrics)

    # 10. Print summary
    print(f"Test set buy signals: {n_signals}, Wins: {n_wins}, Winrate: {winrate:.4f} with Threshold: {optimal_threshold:.4f}")

if __name__ == "__main__":
    main()
