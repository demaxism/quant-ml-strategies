# lstm_predictor origin code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For each point in time, the model uses the past 48 hours of features to **predict the high and low of the next 1 hour.

# Load data
df = pd.read_feather('data/ETH_USDT-1h.feather')


df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
df['change'] = df['close'].pct_change().fillna(0)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 查看前几行
print('start analyze')
print(df.head())

# Normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# Sequence
SEQ_LEN = 48
X, y = [], []
for i in range(len(scaled) - SEQ_LEN - 1):
    X.append(scaled[i:i+SEQ_LEN])
    y.append(scaled[i+SEQ_LEN][1:3])  # predict [high, low]
X, y = np.array(X), np.array(y)

# Train/test split
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

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
print("Chart start date:", plot_dates[0].strftime('%Y-%m-%d'))

# Step 3: Plot with datetime x-axis
plt.figure(figsize=(14, 6))
plt.plot(plot_dates, true[:100, 0], label="True High")
plt.plot(plot_dates, predicted[:100, 0], label="Pred High", linestyle="--")
plt.plot(plot_dates, true[:100, 1], label="True Low")
plt.plot(plot_dates, predicted[:100, 1], label="Pred Low", linestyle="--")

plt.legend()
plt.title("ETH/USDT LSTM Predicted vs True High/Low (1h timeframe)")
plt.xlabel("Datetime")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
