import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import ta

warnings.filterwarnings("ignore")

def load_data(file_path):
    """
    Load data from a feather file.
    """
    data = pd.read_feather(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date').reset_index(drop=True)
    return data

def add_technical_indicators(data):
    """
    Add technical indicators to improve signal quality.
    """
    # Price-based indicators
    data['sma_5'] = data['close'].rolling(window=5).mean()
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['ema_12'] = data['close'].ewm(span=12).mean()
    data['ema_26'] = data['close'].ewm(span=26).mean()
    
    # Volatility
    data['volatility'] = data['close'].rolling(window=20).std()
    data['price_change'] = data['close'].pct_change()
    
    # RSI
    data['rsi'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(data['close'])
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['close'])
    data['bb_high'] = bollinger.bollinger_hband()
    data['bb_low'] = bollinger.bollinger_lband()
    data['bb_mid'] = bollinger.bollinger_mavg()
    data['bb_width'] = (data['bb_high'] - data['bb_low']) / data['bb_mid']
    
    # Volume indicators
    data['volume_sma'] = data['volume'].rolling(window=20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_sma']
    
    # Price position within range
    data['price_position'] = (data['close'] - data['low'].rolling(20).min()) / \
                            (data['high'].rolling(20).max() - data['low'].rolling(20).min())
    
    # Drop NaN values
    data = data.dropna().reset_index(drop=True)
    return data

def preprocess_data(data, seq_len=48, prediction_horizon=4):
    """
    Improved preprocessing with better feature engineering and target definition.
    """
    # Add technical indicators
    data = add_technical_indicators(data)
    
    # Select features
    features = ['open', 'high', 'low', 'close', 'volume',
               'sma_5', 'sma_20', 'ema_12', 'ema_26', 'volatility',
               'price_change', 'rsi', 'macd', 'macd_signal', 'macd_diff',
               'bb_high', 'bb_low', 'bb_mid', 'bb_width',
               'volume_sma', 'volume_ratio', 'price_position']
    
    # Use StandardScaler instead of MinMaxScaler for better gradient flow
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])
    data_scaled = pd.DataFrame(data_scaled, columns=features)
    
    sequences = []
    labels = []
    
    for i in range(len(data_scaled) - seq_len - prediction_horizon):
        seq = data_scaled.iloc[i:i+seq_len].values
        
        # Look ahead multiple hours and require sustained movement
        current_close = data['close'].iloc[i+seq_len]
        future_prices = data['close'].iloc[i+seq_len+1:i+seq_len+prediction_horizon+1]
        
        # More sophisticated labeling: require movement AND sustainability
        max_future_return = (future_prices.max() - current_close) / current_close
        final_return = (future_prices.iloc[-1] - current_close) / current_close
        
        # Stricter criteria: higher threshold + sustained movement
        label = 1 if (max_future_return > 0.015 and final_return > 0.005) else 0
        
        sequences.append(seq)
        labels.append(label)
    
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Positive samples: {np.sum(y)} ({np.mean(y)*100:.2f}%)")
    
    return X, y, scaler

class ImprovedCNNMLP(nn.Module):
    def __init__(self, seq_len, num_features):
        super(ImprovedCNNMLP, self).__init__()
        
        # Multi-scale CNN layers
        self.conv1 = nn.Conv1d(num_features, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=7, padding=3)
        
        # Batch normalization and dropout
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.dropout = nn.Dropout(0.3)
        
        # MLP layers
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(32)
        self.ln3 = nn.LayerNorm(16)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        x = x.permute(0, 2, 1)  # [batch_size, features, seq_len]
        
        # Multi-scale CNN
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.pool(x).squeeze(-1)  # [batch_size, 32]
        
        # MLP layers with residual connections
        residual = x
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.relu(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        
        return x

def build_model(seq_len, num_features):
    """Build the improved model."""
    model = ImprovedCNNMLP(seq_len, num_features)
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=50, patience=10):
    """
    Enhanced training with learning rate scheduling and better monitoring.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_val_f1 = 0
    best_val_precision = 0
    epochs_no_improve = 0
    early_stop = False
    
    train_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    
    for epoch in range(epochs):
        if early_stop:
            print("Early stopping triggered.")
            break
            
        # Training phase
        model.train()
        running_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device).float()
                y_val = y_val.to(device).float()
                
                outputs = model(X_val)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.3).cpu().numpy().astype(int)  # Lower threshold for validation
                
                all_preds.extend(preds.flatten())
                all_labels.extend(y_val.cpu().numpy().flatten())
                all_probs.extend(probs.cpu().numpy().flatten())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        val_accuracies.append(accuracy)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)
        
        # Learning rate scheduling
        scheduler.step()
        
        if epoch % 5 == 0:
            cm = confusion_matrix(all_labels, all_preds)
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {avg_train_loss:.4f} | Val Acc: {accuracy:.4f} | Val Prec: {precision:.4f} | Val Rec: {recall:.4f} | Val F1: {f1:.4f}")
            print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
            print("Confusion Matrix:")
            print(cm)
        
        # Early stopping based on precision (more relevant for trading)
        if precision > best_val_precision and f1 > 0.1:  # Ensure minimum F1
            best_val_precision = precision
            best_val_f1 = f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved! Precision: {precision:.4f}, F1: {f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    history = {
        'train_loss': train_losses,
        'val_accuracy': val_accuracies,
        'val_precision': val_precisions,
        'val_recall': val_recalls,
        'val_f1': val_f1s
    }
    
    return model, history

def advanced_backtest(model, test_loader, threshold_range=np.arange(0.3, 0.9, 0.05)):
    """
    Advanced backtesting with threshold optimization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Collect all predictions and probabilities
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device).float()
            outputs = model(X_test)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(y_test.numpy().flatten())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    best_threshold = 0.5
    best_precision = 0
    best_win_rate = 0
    
    print("Threshold Optimization:")
    print("Threshold | Signals | Hits | Win Rate | Precision | Recall | F1")
    print("-" * 70)
    
    results = []
    for threshold in threshold_range:
        preds = (all_probs > threshold).astype(int)
        
        signals = np.sum(preds)
        hits = np.sum((preds == 1) & (all_labels == 1))
        
        if signals > 0:
            win_rate = hits / signals
            precision = precision_score(all_labels, preds, zero_division=0)
            recall = recall_score(all_labels, preds, zero_division=0)
            f1 = f1_score(all_labels, preds, zero_division=0)
            
            results.append({
                'threshold': threshold,
                'signals': signals,
                'hits': hits,
                'win_rate': win_rate,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            print(f"{threshold:.2f}     | {signals:7} | {hits:4} | {win_rate:7.2%} | {precision:8.3f} | {recall:6.3f} | {f1:.3f}")
            
            # Update best threshold based on precision and minimum signal count
            if precision > best_precision and signals >= 20:  # At least 20 signals
                best_precision = precision
                best_win_rate = win_rate
                best_threshold = threshold
        else:
            print(f"{threshold:.2f}     | {signals:7} | {hits:4} | {0:7.2%} | {0:8.3f} | {0:6.3f} | {0:.3f}")
    
    print("-" * 70)
    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Best Win Rate: {best_win_rate:.2%}")
    print(f"Best Precision: {best_precision:.3f}")
    
    return results, best_threshold

def visualize_results(history, backtest_results):
    """Enhanced visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Training loss
    axes[0, 0].plot(epochs, history['train_loss'])
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    
    # Validation metrics
    axes[0, 1].plot(epochs, history['val_precision'], label='Precision')
    axes[0, 1].plot(epochs, history['val_recall'], label='Recall')
    axes[0, 1].plot(epochs, history['val_f1'], label='F1')
    axes[0, 1].set_title('Validation Metrics')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    
    # Threshold analysis
    if backtest_results:
        thresholds = [r['threshold'] for r in backtest_results]
        win_rates = [r['win_rate'] for r in backtest_results]
        precisions = [r['precision'] for r in backtest_results]
        
        axes[1, 0].plot(thresholds, win_rates, 'b-', label='Win Rate')
        axes[1, 0].plot(thresholds, precisions, 'r-', label='Precision')
        axes[1, 0].set_title('Threshold vs Performance')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].legend()
    
    # Signal count vs threshold
    if backtest_results:
        signals = [r['signals'] for r in backtest_results]
        axes[1, 1].plot(thresholds, signals, 'g-')
        axes[1, 1].set_title('Signal Count vs Threshold')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Number of Signals')
    
    plt.tight_layout()
    plt.savefig('improved_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load Data
    data_path = os.path.join('data', 'BTC_USDT-1h.feather')
    data = load_data(data_path)
    
    # Preprocess Data with improved features
    X, y, scaler = preprocess_data(data, seq_len=48, prediction_horizon=4)
    
    # Temporal split (more realistic for time series)
    split_idx = int(len(X) * 0.7)
    val_split_idx = int(len(X) * 0.85)
    
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:val_split_idx], y[split_idx:val_split_idx]
    X_test, y_test = X[val_split_idx:], y[val_split_idx:]
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Build Model
    seq_len = X.shape[1]
    num_features = X.shape[2]
    model = build_model(seq_len, num_features)
    
    print(f"Model built with input shape: ({seq_len}, {num_features})")
    
    # Calculate class weights more accurately
    pos_weight = torch.tensor([len(y_train) / (2 * np.sum(y_train))], dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define loss, optimizer, and scheduler
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # Train Model
    print("Starting training...")
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=50, patience=15)
    
    # Advanced Backtesting
    print("\nPerforming advanced backtesting...")
    backtest_results, best_threshold = advanced_backtest(model, test_loader)
    
    # Visualize Results
    visualize_results(history, backtest_results)
    
    print(f"\nFinal recommendation: Use threshold {best_threshold:.2f} for trading signals")

if __name__ == "__main__":
    main()