# -*- coding: utf-8 -*-
"""
K线未来N根涨跌预测（XGBoost版）

功能：
1. 数据读取（支持feather/csv）
2. 特征工程（前N根K线的OHLCV及可选特征）
3. 目标变量：未来N根K线内最高价是否较当前收盘价涨TP%以上, 且最低价是否较当前收盘价跌SL%以内
4. XGBoost建模与评估
5. 回测模拟

说明：训练和回测均使用原粒度数据。

作者：专业量化开发者
"""

import random
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import shap
import warnings
import os

_log_print_counter = 0
_log_print_ellipsis_printed = False

def log_print(obj):
    """
    仅前100次调用正常打印，之后只打印一次省略号“...”，其余调用不再打印。
    """
    global _log_print_counter, _log_print_ellipsis_printed
    _log_print_counter += 1
    if _log_print_counter <= 100:
        if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
            lines = str(obj).splitlines()
        elif isinstance(obj, (list, tuple)):
            lines = [str(x) for x in obj]
        else:
            lines = str(obj).splitlines()
        n = len(lines)
        if n <= 20:
            for line in lines:
                print(line)
        else:
            for line in lines[:10]:
                print(line)
            print("...")
            for line in lines[-10:]:
                print(line)
    elif not _log_print_ellipsis_printed:
        print("...")
        _log_print_ellipsis_printed = True
    # else: do nothing

warnings.filterwarnings("ignore")

# ======= 输出信息说明 =======
# prob是模型对“未来涨1%”的概率预测，label是实际是否涨了1%(RISE_THRESHOLD)（回测时已知）

# ====== 用户可调参数 ======
ADVANCED_FEATURES = False  # True: 启用技术指标和K线形态特征；False: 只用基础特征
BET_PROB_THRESHOLD = 0.8   # 下注概率阈值（如0.7表示预测概率大于70%才下注）
RISE_THRESHOLD = 0.01       # 目标变量上涨幅度阈值（如0.01表示1%，可调为0.005等）
FALL_THRESHOLD = -1       # 目标变量下跌幅度阈值（如-0.01表示-1%） 越大越好
FUTURE_K_NUM = 12            # 目标变量观察的未来K线数量（如4表示未来4根K线，可调为3、5等）
LOOKBACK_WINDOW = 20         # 用于特征提取的历史K线数量（如4表示用过去4根K线的特征，可调为3、5等）
TAKE_PROFIT = RISE_THRESHOLD  # 止盈百分比，默认与RISE_THRESHOLD一致
STOP_LOSS = -0.01             # 止损百分比（如-0.01表示-1%止损）
CRYPOTO_CURRENCY = "ETH"  # 可选：指定加密货币（如 "BTC", "ETH", "XRP" 等）
DATA_FILE = f"data/{CRYPOTO_CURRENCY}_USDT-1h.feather"  # 输入数据文件，可选如 "data/ETH_USDT-4h.feather"
trade_pair = DATA_FILE.split('/')[-1].split('-')[0]  # 提取交易对名称，如 "LTC_USDT"

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

def add_features(df, lookback_window=LOOKBACK_WINDOW, bonus=False, advanced=True, rise_threshold=RISE_THRESHOLD, future_k=FUTURE_K_NUM):
    """
    提取特征：前lookback_window根K线的OHLCV、技术指标、K线形态特征
    advanced=True时启用技术指标和K线形态特征，否则只用基础特征
    lookback_window: 用于特征提取的历史K线数量（如4表示用过去4根K线的特征）
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
    for i in range(lookback_window, len(df)-future_k):
        feat = []
        for j in range(lookback_window):
            k_idx = i - lookback_window + j
            # OHLCV
            for col in ['open', 'high', 'low', 'close', 'volume']:
                feat.append(df.iloc[k_idx][col])
                if i == lookback_window:
                    col_names.append(f'{col}_t-{lookback_window-j}')
            if advanced:
                # 技术指标
                for col in ['rsi', 'macd', 'macd_signal', 'macd_hist', 'kdj_k', 'kdj_d', 'kdj_j']:
                    feat.append(df.iloc[k_idx][col])
                    if i == lookback_window:
                        col_names.append(f'{col}_t-{lookback_window-j}')
                # K线形态特征
                open_ = df.iloc[k_idx]['open']
                high_ = df.iloc[k_idx]['high']
                low_ = df.iloc[k_idx]['low']
                close_ = df.iloc[k_idx]['close']
                # 阳线/阴线
                feat.append(is_bullish(open_, close_))
                if i == lookback_window:
                    col_names.append(f'is_bullish_t-{lookback_window-j}')
                # 锤头
                feat.append(is_hammer(open_, high_, low_, close_))
                if i == lookback_window:
                    col_names.append(f'is_hammer_t-{lookback_window-j}')
                # 实体长度
                body = abs(close_ - open_)
                feat.append(body)
                if i == lookback_window:
                    col_names.append(f'body_t-{lookback_window-j}')
                # 上影线比例
                upper = (high_ - max(open_, close_)) / (high_ - low_ + 1e-8)
                feat.append(upper)
                if i == lookback_window:
                    col_names.append(f'upper_shadow_t-{lookback_window-j}')
                # 下影线比例
                lower = (min(open_, close_) - low_) / (high_ - low_ + 1e-8)
                feat.append(lower)
                if i == lookback_window:
                    col_names.append(f'lower_shadow_t-{lookback_window-j}')
            if bonus:
                open_ = df.iloc[k_idx]['open']
                close_ = df.iloc[k_idx]['close']
                # 涨跌幅
                ret = (close_ - open_) / (open_ + 1e-8)
                feat.append(ret)
                if i == lookback_window:
                    col_names.append(f'ret_t-{lookback_window-j}')
        feats.append(feat)
    X = np.array(feats)
    # 目标变量
    y = []
    for i in range(lookback_window, len(df)-future_k):
        cur_close = df.iloc[i]['close']
        future_high = df.iloc[i+1:i+1+future_k]['high'].max()
        future_low = df.iloc[i+1:i+1+future_k]['low'].min()
        rise = (future_high - cur_close) / cur_close
        drawdown = (future_low - cur_close) / cur_close
        label = 1 if (rise >= rise_threshold and drawdown >= FALL_THRESHOLD) else 0
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
    # 平仓方式计数
    cross_count = 0   # 同时穿越止盈止损
    tp_count = 0      # 止盈
    sl_count = 0      # 止损
    none_count = 0    # 未触发止盈止损
    for idx in np.where(bets)[0]:
        if df_test is not None:
            open_date = df_test.iloc[idx]['date'] if 'date' in df_test.columns else ''
        else:
            open_date = ''
        open_price = df_test.iloc[idx]['close'] if (df_test is not None and 'close' in df_test.columns) else 0
        close_price = None
        close_date = ''
        pnl = 0
        # 计算预期止盈止损点
        tp_price = open_price * (1 + take_profit)
        sl_price = open_price * (1 + stop_loss)
        # 止盈止损逻辑
        hit = False
        exit_idx = None
        for k in range(1, future_k+1):
            if idx + k >= len(df_test):
                break
            high = df_test.iloc[idx + k]['high']
            low = df_test.iloc[idx + k]['low']
            tp_price = open_price * (1 + take_profit)
            sl_price = open_price * (1 + stop_loss)
            # 同时穿越止盈止损，按当前K线open价到止盈止损点的距离分配概率
            if high >= tp_price and low <= sl_price:
                cur_open = df_test.iloc[idx + k]['open']
                d_tp = abs(tp_price - cur_open)
                d_sl = abs(sl_price - cur_open)
                total = d_tp + d_sl
                if total == 0:
                    p_tp = 0.5
                else:
                    p_tp = d_sl / total
                if random.random() < p_tp:
                    pnl = take_profit
                    close_price = open_price * (1 + pnl)
                    result_str = "止盈(距离加权, 当前K线open)"
                else:
                    pnl = stop_loss
                    close_price = open_price * (1 + pnl)
                    result_str = "止损(距离加权, 当前K线open)"
                close_date = df_test.iloc[idx + k]['date']
                hit = True
                exit_idx = idx + k
                log_print(f"{idx}\t{open_date}\t{close_date}\t{y_prob[idx]:.4f}\t{y_test[idx]}\t{open_price:.2f}\t{close_price:.2f}\t{pnl:.4f}\t({result_str})")
                log_print(f"  预期止盈: {tp_price:.2f}  预期止损: {sl_price:.2f}  止盈概率: {p_tp:.2f} 止损概率: {1-p_tp:.2f} 当前K线open: {cur_open:.2f}")
                cross_count += 1
                break
            # 止盈
            if high >= tp_price:
                close_price = tp_price
                pnl = take_profit
                close_date = df_test.iloc[idx + k]['date']
                hit = True
                profit_count += 1
                exit_idx = idx + k
                log_print(f"{idx}\t{open_date}\t{close_date}\t{y_prob[idx]:.4f}\t{y_test[idx]}\t{open_price:.2f}\t{close_price:.2f}\t{pnl:.4f}\t(止盈)")
                log_print(f"  预期止盈: {tp_price:.2f}  预期止损: {sl_price:.2f}")
                tp_count += 1
                break
            # 止损
            if low <= sl_price:
                close_price = sl_price
                pnl = stop_loss
                close_date = df_test.iloc[idx + k]['date']
                hit = True
                exit_idx = idx + k
                log_print(f"{idx}\t{open_date}\t{close_date}\t{y_prob[idx]:.4f}\t{y_test[idx]}\t{open_price:.2f}\t{close_price:.2f}\t{pnl:.4f}\t(止损)")
                log_print(f"  预期止盈: {tp_price:.2f}  预期止损: {sl_price:.2f}")
                sl_count += 1
                break
        if not hit:
            # 未触发止盈止损，按最后一根K线close价平仓
            if idx + future_k < len(df_test):
                close_price = df_test.iloc[idx + future_k]['close']
                close_date = df_test.iloc[idx + future_k]['date']
                pnl = (close_price - open_price) / open_price
                exit_idx = idx + future_k
            else:
                close_price = open_price
                close_date = open_date
                pnl = 0
                exit_idx = idx
            log_print(f"{idx}\t{open_date}\t{close_date}\t{y_prob[idx]:.4f}\t{y_test[idx]}\t{open_price:.2f}\t{close_price:.2f}\t{pnl:.4f}\t(未触发止盈止损)")
            log_print(f"  预期止盈: {tp_price:.2f}  预期止损: {sl_price:.2f}")
            none_count += 1
        # 统一打印详细K线信息
        if exit_idx is not None and exit_idx < len(df_test):
            for i in range(idx, exit_idx + 1):
                row = df_test.iloc[i]
                if i == idx:
                    log_print(f"  进场K线: {row['date']} O:{row['open']} C:{row['close']} H:{row['high']} L:{row['low']} V:{row['volume']}")
                elif i == exit_idx:
                    log_print(f"  平仓K线: {row['date']} O:{row['open']} C:{row['close']} H:{row['high']} L:{row['low']} V:{row['volume']}")
                else:
                    log_print(f"  持仓K线: {row['date']} O:{row['open']} C:{row['close']} H:{row['high']} L:{row['low']} V:{row['volume']}")
        trade_pnl.append(pnl)
        equity.append(equity[-1] * (1 + pnl))
    if total_bets > 0:
        profit_ratio = profit_count / total_bets
        log_print(f"止盈交易占比: {profit_count}/{total_bets} = {profit_ratio:.2%}")
    # 输出各种平仓方式的数量和比例
    print("==== 平仓方式统计 ====")
    print(f"同时穿越止盈止损: {cross_count} ({cross_count/total_bets:.2%} of bets)" if total_bets > 0 else f"同时穿越止盈止损: {cross_count} (0.00%)")
    print(f"止盈: {tp_count} ({tp_count/total_bets:.2%} of bets)" if total_bets > 0 else f"止盈: {tp_count} (0.00%)")
    print(f"止损: {sl_count} ({sl_count/total_bets:.2%} of bets)" if total_bets > 0 else f"止损: {sl_count} (0.00%)")
    print(f"未触发止盈止损: {none_count} ({none_count/total_bets:.2%} of bets)" if total_bets > 0 else f"未触发止盈止损: {none_count} (0.00%)")
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
    plt.title(f"Backtest Equity Curve vs {trade_pair} Price")
    plt.tight_layout()
    plt.savefig("equity_curve.png", dpi=200)
    plt.close()
    print("已保存资金曲线图（含价格对比，双y轴）：equity_curve.png")

def main():
    # 1. 加载原始数据
    df = load_data(DATA_FILE)
    print(trade_pair)
    print(f"原数据量: {len(df)}")

    # 2. 特征工程与数据集划分
    X, y, feature_names = add_features(
        df, lookback_window=LOOKBACK_WINDOW, bonus=True, advanced=ADVANCED_FEATURES,
        rise_threshold=RISE_THRESHOLD, future_k=FUTURE_K_NUM
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    date_feat = df['date'].iloc[LOOKBACK_WINDOW:len(df)-FUTURE_K_NUM].reset_index(drop=True)
    train_dates = date_feat.iloc[:len(y_train)]
    test_dates = date_feat.iloc[len(y_train):]
    test_start = test_dates.iloc[0]
    test_end = test_dates.iloc[-1]
    print(f"训练集日期区间: {train_dates.iloc[0]} ~ {train_dates.iloc[-1]}")
    print(f"回测集日期区间: {test_start} ~ {test_end}")

    # 3. 训练模型
    model = train_xgb(X_train, y_train, X_test, y_test)

    # 4. 回测（基于原粒度测试集）
    df_test = df.iloc[LOOKBACK_WINDOW:len(df)-FUTURE_K_NUM].reset_index(drop=True)
    df_test = df_test.iloc[len(y_train):].reset_index(drop=True)
    print(f"回测样本数: {len(y_test)}")
    if len(y_test) == 0:
        print("区间无可用样本，跳过回测。")
    else:
        y_prob, bets, equity, trade_pnl = backtest(
            model, X_test, y_test, df_test=df_test,
            prob_thres=BET_PROB_THRESHOLD, take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, future_k=FUTURE_K_NUM
        )
        # 可选：下注可视化
        try:
            plot_bet_results(df_test, y_prob, bets, n_hist=LOOKBACK_WINDOW)
        except Exception as e:
            print("下注可视化失败：", e)
        # 可选：资金曲线
        try:
            plot_equity_curve(equity, df_test, bets)
        except Exception as e:
            print("资金曲线可视化失败：", e)

    # 可选扩展：SHAP解释（基于训练集）
    try:
        plot_shap(model, X_train, feature_names)
    except Exception as e:
        print("SHAP绘图失败：", e)

if __name__ == "__main__":
    main()
