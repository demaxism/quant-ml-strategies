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

#### 結果

-   リターンを予測しても、そこから再構成した価格は「移動平均のディレイ」みたいなものになってしまう。

-   全然面白くない。

* * * * *

### 💡さらに考えた：**スケール不変な表現**への変換

#### アイデア

-   absolute priceでもreturnでもダメなら、

-   **スケール不変（scale-invariant）かつ局所的に定常な空間に変形してから予測する**方が良いのでは？

#### 試したこと

-   ローカルなスケーリングを行ってからLSTMに入力。

-   最後に元の価格スケールに逆変換して結果を見る。

#### 結果

-   一応うまくnormalizeされたけど、**波形がランダムすぎて今度は予測が安定しない**。

-   これはこれで難しい...

* * * * *

### 🤔現状まとめ

-   ✔ absolute price → NG（スケールがズレるとダメ）

-   ✔ return → NG（ゼロ周辺で予測が意味をなさない）

-   ✔ scale-invariant transformation → 工夫次第で使えそうだけど、今のところ精度微妙

* * * * *

### 🧪 今後の方向性メモ

-   ローカルなスケーリングをもっと工夫？

-   変化率ではなく「パターン」として予測すべきか？

-   MAやボラティリティなど、補助的な指標と一緒に学習させるといいかも？
