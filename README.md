# Predicting Stock Trend using Blending Ensemble

## 📌 Overview

This project predicts stock price direction using an ensemble of machine learning models (blending).

## ⚙️ Models Used

* Linear Regression
* Random Forest
* XGBoost
* Blending Ensemble

## 📊 Features
* data['Open-Close'] = (data.Open - data.Close)/data.Open
* data['High-Low'] = (data.High - data.Low)/data.Low
* data['Daily_Returns'] = np.log(data['Close']/data['Close'].shift(1))
* data['Past_Returns'] = data['Daily_Returns'].shift(1)
* data['ret_5'] = data['Daily_Returns'].rolling(5).mean()
* data['std_5'] = data['Daily_Returns'].rolling(5).std()
* data['Momentum_15'] = data['Close'] - data['Close'].shift(15)
* data['SMA_15'] = data['Close'].rolling(window=15).mean()
* data['EMA_15'] = data['Close'].ewm(span=15,min_periods=15).mean()
* data['Trend'] = np.where(data['Daily_Returns'] > 0, 1, 0)


## 🧪 Methodology

1. Data collection
2. Feature engineering
3. Model training
4. Blending predictions
5. Evaluation

## 📈 Results

* Accuracy: XX%
* Sharpe Ratio: XX
* Backtest Performance included

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/model.py
```

## 📁 Project Structure

(Explain briefly)

## 👤 Author

Pranav Patil,CQF
Email : pranavp222@gmail.com

