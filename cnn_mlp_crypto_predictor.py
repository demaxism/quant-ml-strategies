import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import warnings

warnings.filterwarnings("ignore")

def load_data(file_path):
    """
    Load data from a feather file.
    """
    data = pd.read_feather(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date').reset_index(drop=True)
    return data

def preprocess_data(data, seq_len=24):
    """
    Preprocess the data:
    - MinMaxScaler normalization
    - Create sliding window sequences
    - Define labels
    """
    features = ['open', 'high', 'low', 'close', 'volume']
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[features])
    data_scaled = pd.DataFrame(data_scaled, columns=features)
    
    sequences = []
    labels = []
    for i in range(len(data_scaled) - seq_len - 1):
        seq = data_scaled.iloc[i:i+seq_len].values
        # 计算涨幅是否超过1%
        current_close = data['close'].iloc[i+seq_len]
        next_close = data['close'].iloc[i+seq_len+1]
        pct_change = (next_close - current_close) / current_close
        label = 1 if pct_change > 0.005 else 0
        sequences.append(seq)
        labels.append(label)
    
    X = np.array(sequences)
    y = np.array(labels)
    
    return X, y, scaler

class CNNMLP(nn.Module):
    def __init__(self, seq_len, num_features):
        super(CNNMLP, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # Calculate the output size after Conv1d and MaxPool1d
        conv_output_length = (seq_len - 2) // 2  # (seq_len - kernel_size + 1) / pool_size
        self.fc1 = nn.Linear(64 * conv_output_length, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.5)
            
    def forward(self, x):
        # 调整张量维度从 [batch_size, seq_len, features] 到 [batch_size, features, seq_len]
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def build_model(seq_len, num_features):
    """
    Build the 1D CNN + MLP model.
    """
    model = CNNMLP(seq_len, num_features)
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, patience=5):
    """
    Train the model with EarlyStopping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_f1 = 0
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
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device).float()
                y_val = y_val.to(device).float().unsqueeze(1)
                outputs = model(X_val)
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)
                all_preds.extend(preds.flatten())
                all_labels.extend(y_val.cpu().numpy().flatten())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        val_accuracies.append(accuracy)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)

        # Compute and display confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print(cm)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Accuracy: {accuracy:.4f} | Val Precision: {precision:.4f} | Val Recall: {recall:.4f} | Val F1: {f1:.4f}")
        
        # Early Stopping
        if f1 >= best_f1:
            best_f1 = f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
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

def backtest(model, test_loader, threshold=0.5):
    """
    Perform backtesting on the test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    signals = []
    hits = 0
    total_signals = 0
    
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device).float()
            outputs = model(X_test)
            preds = (torch.sigmoid(outputs) > threshold).cpu().numpy().astype(int)
            for pred, actual in zip(preds, y_test):
                if pred == 1:
                    total_signals += 1
                    if actual == 1:
                        hits += 1
                    signals.append(pred)
    
    if total_signals == 0:
        win_rate = 0
    else:
        win_rate = hits / total_signals
    
    print(f"Backtest Results:")
    print(f"Total Signals: {total_signals}")
    print(f"Hit Count: {hits}")
    print(f"Win Rate: {win_rate:.2%}")
    
    return total_signals, hits, win_rate

def visualize_history(history):
    """
    Visualize training loss and validation metrics.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # Validation F1 Score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_f1'], label='Val F1 Score', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    # Load Data
    data_path = os.path.join('data', 'BTC_USDT-1h.feather')
    data = load_data(data_path)
    
    # Preprocess Data
    X, y, scaler = preprocess_data(data)
    
    # Split Data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, shuffle=True, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.8, shuffle=True, stratify=y_temp)
    
    # Create Datasets and Loaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Build Model
    seq_len = X.shape[1]
    num_features = X.shape[2]
    model = build_model(seq_len, num_features)
    
    # Define Loss and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([26.9]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # Train Model
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, patience=5)
    
    # Visualize Training History
    visualize_history(history)
    
    # Backtest
    backtest(model, test_loader, threshold=0.01)

if __name__ == "__main__":
    main()
