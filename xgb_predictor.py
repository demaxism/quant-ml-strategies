# -*- coding: utf-8 -*-
"""
BTC 4h K线未来4根涨幅预测（XGBoost版）

功能：
1. 数据读取（支持feather/csv）
2. 特征工程（前4根K线的OHLCV及可选特征）
3. 目标变量：未来4根K线内最高价是否较当前收盘价涨1%
4. XGBoost建模与评估
5. 回测模拟
6. 可选：SHAP解释与可视化

作者：专业量化开发者
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import shap
import warnings
import os

warnings.filterwarnings("ignore")

# ======= 输出信息说明 =======
# prob是模型对“未来涨1%”的概率预测，label是实际是否涨了1%(RISE_THRESHOLD)（回测时已知）

# 1. __准确率（Accuracy）__\
#    \= (预测对的样本数) / (总样本数)\
#    反映整体预测的正确性。无论预测为1还是0，只要和实际一致都算对。

# 2. __精确率（Precision）__\
#    \= (真正例 TP) / (预测为正的样本数 TP + FP)\
#    即所有被模型预测为“上涨”（下注）的样本中，实际真的上涨的比例。\
#    —— 关注“下注时命中率”，即下注的质量。

# 3. __召回率（Recall）__\
#    \= (真正例 TP) / (实际为正的样本数 TP + FN)\
#    即所有实际“上涨”的样本中，被模型成功预测出来的比例。\
#    —— 关注“机会捕捉率”，即有多少上涨机会被抓住。

# 4. __命中率（Hit Rate，回测中的命中率）__\
#    \= (下注且实际上涨的次数) / (下注次数)\
#    在本脚本的回测环节，命中率的定义和精确率是一样的：

# ====== 用户可调参数 ======
ADVANCED_FEATURES = False  # True: 启用技术指标和K线形态特征；False: 只用基础特征
BET_PROB_THRESHOLD = 0.75   # 下注概率阈值（如0.7表示预测概率大于70%才下注）
RISE_THRESHOLD = 0.01       # 目标变量上涨幅度阈值（如0.01表示1%，可调为0.005等）
FUTURE_K_NUM = 4            # 目标变量观察的未来K线数量（如4表示未来4根K线，可调为3、5等）
TAKE_PROFIT = RISE_THRESHOLD  # 止盈百分比，默认与RISE_THRESHOLD一致
STOP_LOSS = -0.003             # 止损百分比（如-0.01表示-1%止损）
DATA_FILE = "data/ETH_USDT-4h.feather"  # 输入数据文件，可选如 "data/ETH_USDT-4h.feather"
FINE_DATA_FILE = "data/ETH_USDT-1h.feather"


def load_data(file_path):
    """
    读取K线数据，支持feather和csv，按时间升序排序
    """
    ext = os.path.splitext(file_path)[-1]
    if ext == '.feather':
        df = pd.read_feather(file_path)
    else:
        df = pd.read_csv(file_path)
    # 兼容字段名
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'time' in df.columns:
        df['date'] = pd.to_datetime(df['time'])
    else:
        raise ValueError("找不到时间字段")
    df = df.sort_values('date').reset_index(drop=True)
    return df

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def calc_kdj(df, n=9, k_period=3, d_period=3):
    low_min = df['low'].rolling(window=n, min_periods=1).min()
    high_max = df['high'].rolling(window=n, min_periods=1).max()
    rsv = (df['close'] - low_min) / (high_max - low_min + 1e-8) * 100
    k = rsv.ewm(com=k_period-1, adjust=False).mean()
    d = k.ewm(com=d_period-1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

def is_bullish(open_, close_):
    return int(close_ > open_)

def is_hammer(open_, high_, low_, close_):
    body = abs(close_ - open_)
    upper = high_ - max(open_, close_)
    lower = min(open_, close_) - low_
    return int(body < (high_ - low_) * 0.3 and lower > 2 * body and upper < body)

def add_features(df, n_hist=4, bonus=False, advanced=True, rise_threshold=RISE_THRESHOLD, future_k=FUTURE_K_NUM):
    """
    提取特征：前n_hist根K线的OHLCV、技术指标、K线形态特征
    advanced=True时启用技术指标和K线形态特征，否则只用基础特征
    rise_threshold: 目标变量上涨幅度阈值（如0.01表示1%，可调为0.005等）
    future_k: 目标变量观察的未来K线数量（如4表示未来4根K线）
    """
    df = df.copy()
    if advanced:
        # 计算技术指标
        df['rsi'] = calc_rsi(df['close'])
        macd, macd_signal, macd_hist = calc_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        k, d, j = calc_kdj(df)
        df['kdj_k'] = k
        df['kdj_d'] = d
        df['kdj_j'] = j

    feats = []
    col_names = []
    for i in range(n_hist, len(df)-4):
        feat = []
        for j in range(n_hist):
            k_idx = i - n_hist + j
            # OHLCV
            for col in ['open', 'high', 'low', 'close', 'volume']:
                feat.append(df.iloc[k_idx][col])
                if i == n_hist:
                    col_names.append(f'{col}_t-{n_hist-j}')
            if advanced:
                # 技术指标
                for col in ['rsi', 'macd', 'macd_signal', 'macd_hist', 'kdj_k', 'kdj_d', 'kdj_j']:
                    feat.append(df.iloc[k_idx][col])
                    if i == n_hist:
                        col_names.append(f'{col}_t-{n_hist-j}')
                # K线形态特征
                open_ = df.iloc[k_idx]['open']
                high_ = df.iloc[k_idx]['high']
                low_ = df.iloc[k_idx]['low']
                close_ = df.iloc[k_idx]['close']
                # 阳线/阴线
                feat.append(is_bullish(open_, close_))
                if i == n_hist:
                    col_names.append(f'is_bullish_t-{n_hist-j}')
                # 锤头
                feat.append(is_hammer(open_, high_, low_, close_))
                if i == n_hist:
                    col_names.append(f'is_hammer_t-{n_hist-j}')
                # 实体长度
                body = abs(close_ - open_)
                feat.append(body)
                if i == n_hist:
                    col_names.append(f'body_t-{n_hist-j}')
                # 上影线比例
                upper = (high_ - max(open_, close_)) / (high_ - low_ + 1e-8)
                feat.append(upper)
                if i == n_hist:
                    col_names.append(f'upper_shadow_t-{n_hist-j}')
                # 下影线比例
                lower = (min(open_, close_) - low_) / (high_ - low_ + 1e-8)
                feat.append(lower)
                if i == n_hist:
                    col_names.append(f'lower_shadow_t-{n_hist-j}')
            if bonus:
                open_ = df.iloc[k_idx]['open']
                close_ = df.iloc[k_idx]['close']
                # 涨跌幅
                ret = (close_ - open_) / (open_ + 1e-8)
                feat.append(ret)
                if i == n_hist:
                    col_names.append(f'ret_t-{n_hist-j}')
        feats.append(feat)
    X = np.array(feats)
    # 目标变量
    y = []
    for i in range(n_hist, len(df)-future_k):
        cur_close = df.iloc[i]['close']
        future_high = df.iloc[i+1:i+1+future_k]['high'].max()
        label = 1 if (future_high - cur_close) / cur_close >= rise_threshold else 0
        y.append(label)
    y = np.array(y)
    return X, y, col_names

def train_xgb(X_train, y_train, X_test, y_test):
    """
    训练XGBoost分类器，输出准确率、精确率、召回率
    """
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    print(f"准确率: {acc:.4f}  精确率: {prec:.4f}  召回率: {rec:.4f}")
    return model

def backtest(model, X_test, y_test, df_test=None, prob_thres=0.7, take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, future_k=FUTURE_K_NUM):
    """
    回测：预测概率>prob_thres视为下注，统计命中率，并输出每次下注的信息
    新增：资金曲线，止盈止损机制
    """
    y_prob = model.predict_proba(X_test)[:,1]
    bets = y_prob > prob_thres
    total_bets = bets.sum()
    hits = ((y_test == 1) & bets).sum()
    hit_rate = hits / total_bets if total_bets > 0 else 0
    print(f"回测：下注次数={total_bets} 命中次数={hits} 命中率={hit_rate:.2%}")

    # 盈亏与资金曲线
    equity = [1.0]  # 初始资金为1
    trade_pnl = []  # 每笔盈亏
    if df_test is not None:
        print("每次下注详情：")
        print("idx\topen_date\tclose_date\tprob\tlabel\topen\tclose\tpnl")
    profit_count = 0  # 止盈次数统计
    for idx in np.where(bets)[0]:
        if df_test is not None:
            open_date = df_test.iloc[idx]['date'] if 'date' in df_test.columns else ''
        else:
            open_date = ''
        open_price = df_test.iloc[idx]['close'] if (df_test is not None and 'close' in df_test.columns) else 0
        close_price = None
        close_date = ''
        pnl = 0
        # 止盈止损逻辑
        hit = False
        for k in range(1, future_k+1):
            if idx + k >= len(df_test):
                break
            high = df_test.iloc[idx + k]['high']
            low = df_test.iloc[idx + k]['low']
            tp_price = open_price * (1 + take_profit)
            sl_price = open_price * (1 + stop_loss)
            # 同时穿越止盈止损，采用中间法（平均法）
            if high >= tp_price and low <= sl_price:
                pnl = (take_profit + stop_loss) / 2
                close_price = open_price * (1 + pnl)
                close_date = df_test.iloc[idx + k]['date']
                hit = True
                print(f"{idx}\t{open_date}\t{close_date}\t{y_prob[idx]:.4f}\t{y_test[idx]}\t{open_price:.2f}\t{close_price:.2f}\t{pnl:.4f}\t(平均法成交)")
                break
            # 止盈
            if high >= tp_price:
                close_price = tp_price
                pnl = take_profit
                close_date = df_test.iloc[idx + k]['date']
                hit = True
                profit_count += 1
                print(f"{idx}\t{open_date}\t{close_date}\t{y_prob[idx]:.4f}\t{y_test[idx]}\t{open_price:.2f}\t{close_price:.2f}\t{pnl:.4f}\t(止盈)")
                break
            # 止损
            if low <= sl_price:
                close_price = sl_price
                pnl = stop_loss
                close_date = df_test.iloc[idx + k]['date']
                hit = True
                print(f"{idx}\t{open_date}\t{close_date}\t{y_prob[idx]:.4f}\t{y_test[idx]}\t{open_price:.2f}\t{close_price:.2f}\t{pnl:.4f}\t(止损)")
                break
        if not hit:
            # 未触发止盈止损，按最后一根K线close价平仓
            if idx + future_k < len(df_test):
                close_price = df_test.iloc[idx + future_k]['close']
                close_date = df_test.iloc[idx + future_k]['date']
                pnl = (close_price - open_price) / open_price
            else:
                close_price = open_price
                close_date = open_date
                pnl = 0
            print(f"{idx}\t{open_date}\t{close_date}\t{y_prob[idx]:.4f}\t{y_test[idx]}\t{open_price:.2f}\t{close_price:.2f}\t{pnl:.4f}\t(未触发止盈止损)")
        trade_pnl.append(pnl)
        equity.append(equity[-1] * (1 + pnl))
    if total_bets > 0:
        profit_ratio = profit_count / total_bets
        print(f"止盈交易占比: {profit_count}/{total_bets} = {profit_ratio:.2%}")
    equity = np.array(equity)
    return y_prob, bets, equity, trade_pnl

def plot_shap(model, X, feature_names):
    """
    可选：画出SHAP特征重要性
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=200)
    plt.close()
    print("已保存SHAP特征重要性图：shap_summary.png")

def plot_bet_results(df, y_prob, bets, n_hist=4):
    """
    可选：画出下注时机与实际涨幅的可视化
    """
    bet_idx = np.where(bets)[0]
    plt.figure(figsize=(14,6))
    plt.plot(df['date'][n_hist:len(df)-4], df['close'][n_hist:len(df)-4], label='Close')
    plt.scatter(df['date'][n_hist:len(df)-4].iloc[bet_idx], df['close'][n_hist:len(df)-4].iloc[bet_idx], 
                c='r', marker='^', label='Bet', s=60)
    plt.title("下注时机与收盘价")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bet_visualization.png", dpi=200)
    plt.close()
    print("已保存下注可视化图：bet_visualization.png")

def plot_equity_curve(equity, df_test, bets):
    """
    绘制资金曲线（右轴，原始数值），叠加价格曲线（左轴，原始），横轴为日期
    """
    import numpy as np
    fig, ax1 = plt.subplots(figsize=(12,6))
    # 价格曲线（左轴）
    ax1.plot(df_test['date'], df_test['close'], label="Price Curve", color='orange', alpha=0.5)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price", color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')
    # 资金曲线（右轴，原始数值）
    ax2 = ax1.twinx()
    trade_idx = np.where(bets)[0]
    trade_dates = df_test['date'].iloc[trade_idx]
    equity_arr = np.array(equity[1:])
    ax2.plot(trade_dates, equity_arr, label="Equity Curve", color='b')
    ax2.set_ylabel("Equity", color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title("Backtest Equity Curve vs Price")
    plt.tight_layout()
    plt.savefig("equity_curve.png", dpi=200)
    plt.close()
    print("已保存资金曲线图（含价格对比，双y轴）：equity_curve.png")

def main():
    # 1. 训练阶段（4h数据）
    df_train = load_data(DATA_FILE)
    print(f"训练数据量: {len(df_train)}")

    n_hist = 4
    X, y, feature_names = add_features(
        df_train, n_hist=n_hist, bonus=True, advanced=ADVANCED_FEATURES,
        rise_threshold=RISE_THRESHOLD, future_k=FUTURE_K_NUM
    )
    print(f"训练特征维度: {X.shape}, 正样本比例: {y.mean():.2%}")

    # 时间序列分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    total = len(y)
    train_size = len(y_train)
    test_size = len(y_test)
    date_feat = df_train['date'].iloc[n_hist:len(df_train)-FUTURE_K_NUM].reset_index(drop=True)
    train_dates = date_feat.iloc[:train_size]
    test_dates = date_feat.iloc[train_size:]
    print(f"训练集日期区间: {train_dates.iloc[0]} ~ {train_dates.iloc[-1]}")
    print(f"回测集（4h）日期区间: {test_dates.iloc[0]} ~ {test_dates.iloc[-1]}")
    print(f"回测集区间价格始值: {df_train['close'].iloc[test_dates.index[0]]}")
    print(f"回测集区间价格终值: {df_train['close'].iloc[test_dates.index[-1]]}")

    # 训练模型
    model = train_xgb(X_train, y_train, X_test, y_test)

    # 2. 回测阶段（1h数据，粒度更细，时间段不重叠）
    df_fine = load_data(FINE_DATA_FILE)
    print(f"回测数据量（细粒度）: {len(df_fine)}")

    # 取回测起始时间（4h训练集最后一根K线的date之后）
    last_train_date = train_dates.iloc[-1]
    df_fine_bt = df_fine[df_fine['date'] > last_train_date].reset_index(drop=True)
    print(f"细粒度回测区间: {df_fine_bt['date'].iloc[0]} ~ {df_fine_bt['date'].iloc[-1]}")

    # 预测窗口放大（如4*4=16根1h）
    fine_future_k = FUTURE_K_NUM * 4  # 假设4h:1h=4:1
    X_fine, y_fine, feature_names_fine = add_features(
        df_fine_bt, n_hist=n_hist, bonus=True, advanced=ADVANCED_FEATURES,
        rise_threshold=RISE_THRESHOLD, future_k=fine_future_k
    )
    print(f"细粒度回测特征维度: {X_fine.shape}, 正样本比例: {y_fine.mean():.2%}")

    # 对齐df_test长度
    df_test = df_fine_bt.iloc[n_hist:len(df_fine_bt)-fine_future_k].reset_index(drop=True)
    print(f"df_test shape: {df_test.shape}, X_fine shape: {X_fine.shape}, y_fine shape: {y_fine.shape}")

    # 检查shape一致性
    if not (len(df_test) == X_fine.shape[0] == y_fine.shape[0]):
        print("警告：df_test、X_fine、y_fine长度不一致，请检查数据对齐！")
        min_len = min(len(df_test), X_fine.shape[0], y_fine.shape[0])
        df_test = df_test.iloc[:min_len]
        X_fine = X_fine[:min_len]
        y_fine = y_fine[:min_len]

    # 回测
    y_prob, bets, equity, trade_pnl = backtest(
        model, X_fine, y_fine, df_test=df_test,
        prob_thres=BET_PROB_THRESHOLD, take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, future_k=fine_future_k
    )

    # 可选扩展：SHAP解释
    try:
        plot_shap(model, X_train, feature_names)
    except Exception as e:
        print("SHAP绘图失败：", e)

    # 可选扩展：下注时机可视化
    try:
        plot_bet_results(df_test, y_prob, bets, n_hist=n_hist)
    except Exception as e:
        print("下注可视化失败：", e)

    # 资金曲线可视化
    try:
        plot_equity_curve(equity, df_test, bets)
    except Exception as e:
        print("资金曲线可视化失败：", e)

if __name__ == "__main__":
    main()
