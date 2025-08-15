# Price Prediction ML Strategy

This repository is a personal learning project focused on predicting financial prices using various machine learning techniques. The project explores and implements models such as:

- **LSTM (Long Short-Term Memory networks)**
- **XGBoost (Extreme Gradient Boosting)**
- **CNN-MLP (Convolutional Neural Network with Multi-Layer Perceptron)**

The goal is to experiment with and compare these approaches for time series forecasting and price prediction tasks. This project serves as a hands-on exploration of modern ML methods in quantitative finance.


📈 LSTMで仮想通貨価格を予測しようとしてハマった話
----------------------------

### 背景

LSTMでBTCの高値・安値を予測しようとしたが、うまくいかない点がいくつかあったので、試行錯誤した内容をメモ。

* * * * *

### ❗問題1：予測値が全体的に低すぎる

#### 現象

テスト期間に入ると、LSTMの予測値がずっと低く出る。

#### 原因

-   トレーニングデータのBTC価格がテスト期間よりもかなり安い。

-   LSTMはトレーニング期間のスケールを学習してしまい、それ以上に「跳ねる」ことができない。

-   結果として、全体的に予測値が「過去の価格帯に引っ張られる」。

  
  <img width="1400" height="600" alt="lstm_predictions_ETH_USDT_4h_turn5" src="https://github.com/user-attachments/assets/10aa7952-f39b-4bc6-b207-f68111ed82c9" />
 [この図では、予測価格が実際の価格よりやや高めに出ている]

 <img width="1400" height="600" alt="lstm_predictions_BTC_USDT_4h_turn1" src="https://github.com/user-attachments/assets/91106e05-194e-49bd-af0e-f4b08bdfc773" />
 [そしてこちらの図では、予測価格が完全に外れてしまっている…]

#### 対策

👉 価格そのものではなく「変化率（リターン）」を予測するように変更。

* * * * *

### ✔試してみた：リターン（percentage change）での予測

#### そもそも「percentage change」って？

いわゆる**変化率**のことで、以下のように計算する：

`(current_price - previous_price) / previous_price`

つまり、前回の価格に対してどれだけ上下したかを％で表したもの。\
価格そのものではなく、**上下の"動き"に注目する**ため、価格のスケールに左右されにくいのがメリット。

#### 結果

-   グラフはかなり改善された。

-   スケールの変動に強くなった感はある。


<img width="1400" height="600" alt="lstm_predictions_BTC_USDT_4h_turn1 (1)" src="https://github.com/user-attachments/assets/b6641f58-c130-45f4-a85e-d193fecfccc6" />


* * * * *

### ❗問題2：リターンも意外とダメ

#### 現象

-   リターンはゼロ付近でランダムに動く値。

-   LSTMで予測すると「ほぼ0の直線」になってしまい、意味のない予測になる。

  <img width="1100" height="300" alt="Figure_1-scale-invariant" src="https://github.com/user-attachments/assets/085c629e-d586-4441-b9f7-977dcda8f4d0" />


#### 結果

-   リターンを予測しても、そこから再構成した価格は「移動平均のディレイ」みたいなものになってしまう。

-   全然面白くない。

* * * * *

### 🤔現状まとめ

-   ✔ absolute price → NG（スケールがズレるとダメ）

-   ✔ return → NG（ゼロ周辺で予測が意味をなさない）

* * * * *

### 🧪 今後の方向性メモ

-   scale-invariant transformation， ローカルなスケーリングをもっと工夫？

-   変化率ではなく「パターン」として予測すべきか？

-   MAやボラティリティなど、補助的な指標と一緒に学習させるといいかも？


🎲 LSTMで仮想通貨のリターンを予測してみた話（抽選ガチャ方式）
----------------------------------

### 🧠 モデル概要

まず、使っているLSTMモデルはこんな感じ：

`class LSTMPriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(6, 64, LAYERS, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])`

トレーニングは以下のように早期終了（Early Stopping）つきで行う：

-   エポックごとに損失（MSE）を評価

-   前回より良くならなかったらカウントアップ

-   3エポック連続で改善なし→終了

`# Early stopping logic
if best_loss - avg_loss > min_delta:
    best_loss = avg_loss
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}...")
        break`

* * * * *

### 🎰 抽選方式のトレーニング＆テスト

このトレーニングを**ランダムに100回くらい**繰り返して、\
それぞれのモデルでバックテスト（回測）を実施。\
つまり、**「当たりモデル」を探すガチャ方式！**

* * * * *

### 📈 最初に出た夢のような結果

-   あるモデルで年利**10倍**のトレード結果が出た！

-   それがこのグラフ：\
    <img width="1200" height="500" alt="lstm_equity_curve_realtime_ETC_USDT_2025-08-07_21-27-16" src="https://github.com/user-attachments/assets/9fc0a91a-f4bd-4755-a108-f52214e1cffb" />


...すごくワクワクした。

* * * * *

### 😰 でも回測ロジックにバグがあった

実は、**ローソク足の高値と安値が同時にTP（利確）とSL（損切）を両方満たす**場合に、exit価格を「TPとSLの平均」で処理してた。

でもそれ、よく考えたら変だよね？

#### 修正内容：

-   exit価格は**open価格で処理**するように変更（保守的で統計的にも自然）

#### 修正後のグラフ：

<img width="1200" height="500" alt="lstm_equity_curve_realtime_ETH_USDT_2025-08-08_23-25-37" src="https://github.com/user-attachments/assets/6faa591b-d389-4392-8f3e-9e59dff86880" />


現実的な利益率に下がったけど、まぁまぁいい感じ。

* * * * *

### 💸 手数料が地味に致命的だった

さらに深掘りすると、問題がもう1つ...。

-   確かに年利100%とか出てるけど、**取引回数が数千回**

-   しかも**取引手数料（0.05%〜0.1%）**は全く考慮してなかった

-   冷静に考えると、**1回のトレードで得られる利益が手数料すらカバーできてない**ことが多い

-   チャ方式で何度もトレーニングして得られた高いバックテスト収益は、検証データへのoverfitting（オーバーフィッティング）である可能性が高い。

エントリーの閾値を上げて取引回数を減らしながら、やっとの思いでこの収益曲線を出せた。前に出たようなワクワクする収益曲線ではないけど、現実的で妥当な結果ではある。
<img width="1200" height="500" alt="lstm_equity_curve_realtime_ETH_USDT_2025-08-10_11-19-06" src="https://github.com/user-attachments/assets/3dc16121-343f-4cf3-938e-3aa63d400de6" />


* * * * *

### 🔚 結論と次のTODO

-   ✔ 年利だけ見て喜ぶのは早すぎた

-   ✔ 精度も大事だけど「手数料含めた実益」が大事

-   🚧 今後は：

    -   取引コストをちゃんと加味した回測ロジックにする

    -   トレード頻度を減らして、1回のトレードの利益を大きくする方向に最適化する

    -   精度を上げるだけでなく、**利益率＞手数料**を重視する指標を導入する

* * * * *

LSTMでトレードやるの、なかなか奥が深い...
