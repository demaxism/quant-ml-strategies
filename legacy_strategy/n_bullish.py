import pandas as pd
import os
import argparse

# === 策略参数 ===
N_BULLISH_COUNT = 4              # 连续阳线数量触发观察
INITIAL_CASH = 1000.0            # 初始资金
ORDER_SIZE_RATIO = 0.95           # 每次交易使用资金比例
TP_PERCENT = 0.10                # 止盈 +10%
SL_PERCENT = -0.10               # 止损 -10%
REBOUND_LOWER = 0.93             # 反弹观察下限
REBOUND_EXIT = 0.97              # 反弹平仓上限

# === 回测时间范围设置 ===
BACKTEST_START = "2020-12-15"
BACKTEST_END = "2024-03-01"

# === 数据加载与预处理 ===
parser = argparse.ArgumentParser(description='Scale-invariant transformation for price data')
parser.add_argument('--datafile', type=str, default='data/ETH_USDT-1h.feather',
    help='Path to input feather file (e.g., data/ETH_USDT-4h.feather)')
parser.add_argument('--bt_from', type=str, default='2020-12-15',
    help='Backtest start date (e.g., 2025-03-01)')
parser.add_argument('--bt_to', type=str, default='2024-03-01',
    help='Backtest end date (e.g., 2025-03-01)')
args = parser.parse_args()

data_src = args.datafile
BACKTEST_START = args.bt_from
BACKTEST_END = args.bt_to
# Get filename only: "BTC_USDT-1h.feather"
filename = os.path.basename(data_src)
# Extract currency pair: "BTC_USDT"
pair = filename.split('-')[0]
print(pair)  # Output: BTC_USDT
df = pd.read_feather(data_src)  # 你的K线数据路径

print('start analyze')
print(df.head())

df['date'] = pd.to_datetime(df['date'])
df['is_bullish'] = df['close'] > df['open']

# === 根据时间范围过滤数据 ===
df = df[(df['date'] >= BACKTEST_START) & (df['date'] <= BACKTEST_END)].reset_index(drop=True)

# 输出时间范围和价格
start_time = df['date'].min()
end_time = df['date'].max()
print(f"\n📅 Time range in dataset: {start_time} → {end_time}")
start_open = df.iloc[0]['open']
end_close = df.iloc[-1]['close']
print(f"💰 Start Open Price: {start_open:.2f}")
print(f"💰 End Close Price: {end_close:.2f}")

print("\n===== Back test =====")

# === 状态变量初始化 ===
cash = INITIAL_CASH
position = None
observing = True
trades = []

# === 主回测循环 ===
idx = 0
while idx < len(df):
    # --- 阶段一：观察并挂单 ---
    if observing and position is None:
        if idx >= N_BULLISH_COUNT:
            recent = df.iloc[idx - N_BULLISH_COUNT:idx]
            if recent['is_bullish'].all():
                buy_price = recent.iloc[-1]['close']
                size = (cash * ORDER_SIZE_RATIO) / buy_price
                position = {
                    'entry_price': buy_price,
                    'size': size,
                    'entry_time': df.at[idx, 'date'],
                    'in_rebound_watch': False
                }
                cash -= buy_price * size
                print(f"[{df.at[idx, 'date']}] Market Buy at {buy_price:.2f}, size={size:.4f}, cash={cash:.2f}")
                observing = False

    # --- 阶段三：持仓管理 ---
    elif position:
        current_close = df.at[idx, 'close']
        current_high = df.at[idx, 'high']
        entry_price = position['entry_price']

        # 止盈止损
        if current_high >= entry_price * (1 + TP_PERCENT):
            exit_price = current_high
            pnl = (exit_price - entry_price) * position['size']
            cash += exit_price * position['size']
            print(f"[{df.at[idx, 'date']}] TP Hit: entry={entry_price:.2f}, exit={exit_price:.2f}, pnl={pnl:.2f}, cash={cash:.2f}")
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': df.at[idx, 'date'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': position['size'],
                'pnl': pnl,
                'memo': 'TP'
            })
            position = None
            observing = True
        elif position['in_rebound_watch'] and current_high >= entry_price * REBOUND_EXIT:
            exit_price = current_high
            pnl = (exit_price - entry_price) * position['size']
            cash += exit_price * position['size']
            print(f"[{df.at[idx, 'date']}] Rebound Exit: entry={entry_price:.2f}, exit={exit_price:.2f}, pnl={pnl:.2f}, cash={cash:.2f}")
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': df.at[idx, 'date'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': position['size'],
                'pnl': pnl,
                'memo': 'rebound_exit'
            })
            position = None
            observing = True
        elif current_close <= entry_price * (1 + SL_PERCENT):
            exit_price = current_close
            pnl = (exit_price - entry_price) * position['size']
            cash += exit_price * position['size']
            print(f"[{df.at[idx, 'date']}] SL Hit: entry={entry_price:.2f}, exit={exit_price:.2f}, pnl={pnl:.2f}, cash={cash:.2f}")
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': df.at[idx, 'date'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': position['size'],
                'pnl': pnl,
                'memo': 'SL'
            })
            position = None
            observing = True
        else:
            # 反弹观察
            if not position['in_rebound_watch'] and current_close <= entry_price * REBOUND_LOWER:
                position['in_rebound_watch'] = True
                print(f"[{df.at[idx, 'date']}] Enter Rebound Watch (price={current_close:.2f})")

    idx += 1

# === 回测结束后强制平仓所有持仓 ===
if position:
    exit_price = df.iloc[-1]['close']
    pnl = (exit_price - position['entry_price']) * position['size']
    cash += exit_price * position['size']
    print(f"[{df.iloc[-1]['date']}] Forced Exit at End: entry={position['entry_price']:.2f}, exit={exit_price:.2f}, pnl={pnl:.2f}, cash={cash:.2f}")
    trades.append({
        'entry_time': position['entry_time'],
        'exit_time': df.iloc[-1]['date'],
        'entry_price': position['entry_price'],
        'exit_price': exit_price,
        'size': position['size'],
        'pnl': pnl,
        'memo': 'forced_exit'
    })
    position = None

# === 统计与输出 ===
total_trades = len(trades)
winning_trades = [t for t in trades if t['pnl'] > 0]
losing_trades = [t for t in trades if t['pnl'] <= 0]
total_pnl = sum(t['pnl'] for t in trades)
rebound_exit = [t for t in trades if t['memo'] == 'rebound_exit']

print("\n===== Summary =====")
print(f"Total Trades       : {total_trades}")
print(f"Winning Trades     : {len(winning_trades)}")
print(f"Losing Trades      : {len(losing_trades)}")
print(f"Rebound TP         : {len(rebound_exit)}")
print(f"Total PnL (USDT)   : {total_pnl:.2f}")
print(f"Final Cash (USDT)  : {cash:.2f}")


import matplotlib.pyplot as plt

# 1. 准备 BTC 价格
price_dates = df['date']
price_series = df['close']

# 2. 生成现金曲线（现金变化点）
cash_curve = []
cash_point_idx = 0
current_cash = INITIAL_CASH

for i, row in df.iterrows():
    dt = row['date']
    cp = row['close']
    if cash_point_idx < len(trades):
        t = trades[cash_point_idx]
        if dt >= t['exit_time']:
            current_cash += (t['exit_price'] * t['size']) - (t['entry_price'] * t['size'])
            cash_point_idx += 1
    cash_curve.append(current_cash)

# 3. 交易点
buy_times = [t['entry_time'] for t in trades]
buy_prices = [t['entry_price'] for t in trades]
sell_times = [t['exit_time'] for t in trades]
sell_prices = [t['exit_price'] for t in trades]

# === 绘图 ===
fig, ax1 = plt.subplots(figsize=(14, 6))

# 左边 Y 轴：BTC 价格
ax1.plot(price_dates, price_series, label=f'{pair} Close Price', color='orange', alpha=0.4)
ax1.scatter(buy_times, buy_prices, marker='^', color='green', label='Buy', zorder=5)
ax1.scatter(sell_times, sell_prices, marker='v', color='red', label='Sell', zorder=5)
ax1.set_ylabel(f'{pair} Price (USDT)', color='black')
ax1.tick_params(axis='y', labelcolor='black')

# 右边 Y 轴：现金曲线
ax2 = ax1.twinx()
ax2.plot(price_dates, cash_curve, label='Cash (USDT)', color='blue', linewidth=2)
ax2.set_ylabel('Cash (USDT)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# 标题、网格、图例
plt.title(f"Backtest Result: {pair} Price vs Cash with Buy/Sell Points")
ax1.grid(True)

# 合并图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.tight_layout()
plt.show()
