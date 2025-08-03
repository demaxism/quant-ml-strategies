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
# prob是模型对“未来涨1%”的概率预测，label是实际是否涨了1%（回测时已知）

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
BET_PROB_THRESHOLD = 0.85   # 下注概率阈值（如0.7表示预测概率大于70%才下注）
RISE_THRESHOLD = 0.01       # 目标变量上涨幅度阈值（如0.01表示1%，可调为0.005等）
DATA_FILE = "data/LTC_USDT-4h.feather"  # 输入数据文件，可选如 "data/ETH_USDT-1h.feather"


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

def add_features(df, n_hist=4, bonus=False, advanced=True, rise_threshold=RISE_THRESHOLD):
    """
    提取特征：前n_hist根K线的OHLCV、技术指标、K线形态特征
    advanced=True时启用技术指标和K线形态特征，否则只用基础特征
    rise_threshold: 目标变量上涨幅度阈值（如0.01表示1%，可调为0.005等）
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
    for i in range(n_hist, len(df)-4):
        cur_close = df.iloc[i]['close']
        future_high = df.iloc[i+1:i+5]['high'].max()
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

def backtest(model, X_test, y_test, df_test=None, prob_thres=0.7):
    """
    回测：预测概率>prob_thres视为下注，统计命中率，并输出每次下注的信息
    """
    y_prob = model.predict_proba(X_test)[:,1]
    bets = y_prob > prob_thres
    total_bets = bets.sum()
    hits = ((y_test == 1) & bets).sum()
    hit_rate = hits / total_bets if total_bets > 0 else 0
    print(f"回测：下注次数={total_bets} 命中次数={hits} 命中率={hit_rate:.2%}")

    # 输出每次下注的信息
    if df_test is not None:
        print("每次下注详情：")
        print("idx\tdate\t\tprob\tlabel\tclose")
        for idx in np.where(bets)[0]:
            date = df_test.iloc[idx]['date'] if 'date' in df_test.columns else ''
            close = df_test.iloc[idx]['close'] if 'close' in df_test.columns else ''
            print(f"{idx}\t{date}\t{y_prob[idx]:.4f}\t{y_test[idx]}\t{close}")
    return y_prob, bets

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

def main():
    # 1. 数据读取
    df = load_data(DATA_FILE)
    print(f"数据量: {len(df)}")

    # 2. 特征工程
    n_hist = 4
    X, y, feature_names = add_features(df, n_hist=n_hist, bonus=True, advanced=ADVANCED_FEATURES, rise_threshold=RISE_THRESHOLD)  # bonus=True可选扩展
    print(f"特征维度: {X.shape}, 正样本比例: {y.mean():.2%}")

    # 3. 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # 时间序列不打乱
    )

    # 输出训练集和回测集的日期区间
    total = len(y)
    train_size = len(y_train)
    test_size = len(y_test)
    # 特征和标签起始于 df[n_hist:len(df)-4]
    date_feat = df['date'].iloc[n_hist:len(df)-4].reset_index(drop=True)
    train_dates = date_feat.iloc[:train_size]
    test_dates = date_feat.iloc[train_size:]
    print(f"训练集日期区间: {train_dates.iloc[0]} ~ {train_dates.iloc[-1]}")
    print(f"回测集日期区间: {test_dates.iloc[0]} ~ {test_dates.iloc[-1]}")
    print(f"回测集区间价格始值: {df['close'].iloc[test_dates.index[0]]}")
    print(f"回测集区间价格终值: {df['close'].iloc[test_dates.index[-1]]}")

    # 4. 训练模型
    model = train_xgb(X_train, y_train, X_test, y_test)

    # 5. 回测模拟
    # 获取测试集对应的df行
    df_test = df.iloc[-len(y):].iloc[-len(y_test):].reset_index(drop=True)
    y_prob, bets = backtest(model, X_test, y_test, df_test=df_test, prob_thres=BET_PROB_THRESHOLD)

    # 6. 可选扩展：SHAP解释
    try:
        plot_shap(model, X_train, feature_names)
    except Exception as e:
        print("SHAP绘图失败：", e)

    # 7. 可选扩展：下注时机可视化
    try:
        plot_bet_results(df.iloc[-len(y):], y_prob, bets, n_hist=n_hist)
    except Exception as e:
        print("下注可视化失败：", e)

if __name__ == "__main__":
    main()
