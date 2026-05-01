"""
data.py
-------
Data downloading and feature engineering for the blending ensemble classifier.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def download_data(ticker: str = "TSLA", start: str = "2019-01-05", end: str = "2024-01-05") -> pd.DataFrame:
    """Download historical OHLCV data from Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end)
    data.columns = data.columns.get_level_values(0)  # fix MultiIndex
    return data


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Construct technical features and the binary trend target.

    Features
    --------
    Open-Close     : Intraday open-to-close return
    High-Low       : Intraday range normalised by Low
    Daily_Returns  : Log daily return
    Past_Returns   : Prior day log return
    ret_5          : 5-day rolling mean of log returns
    std_5          : 5-day rolling std of log returns
    Momentum_15    : 15-day price momentum
    SMA_15         : 15-day simple moving average
    EMA_15         : 15-day exponential moving average
    Trend          : 1 if Daily_Returns > 0, else 0  (target)
    """
    df = data.copy()

    df["Open-Close"] = (df.Open - df.Close) / df.Open
    df["High-Low"] = (df.High - df.Low) / df.Low
    df["Daily_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Past_Returns"] = df["Daily_Returns"].shift(1)
    df["ret_5"] = df["Daily_Returns"].rolling(5).mean()
    df["std_5"] = df["Daily_Returns"].rolling(5).std()
    df["Momentum_15"] = df["Close"] - df["Close"].shift(15)
    df["SMA_15"] = df["Close"].rolling(window=15).mean()
    df["EMA_15"] = df["Close"].ewm(span=15, min_periods=15).mean()
    df["Trend"] = np.where(df["Daily_Returns"] > 0, 1, 0)

    df.dropna(inplace=True)
    df.drop(["Open", "High", "Low", "Close", "Volume"], axis=1, inplace=True)
    return df


def fix_multicollinearity(df: pd.DataFrame) -> pd.DataFrame:
    """Drop features with high VIF (>10) identified during EDA."""
    cols_to_drop = ["High-Low", "std_5", "SMA_15", "EMA_15"]
    return df.drop(cols_to_drop, axis=1)


def get_feature_target(df: pd.DataFrame):
    """Return X (features) and y (target) arrays."""
    feature_cols = ["Open-Close", "Daily_Returns", "Past_Returns", "ret_5"]
    X = df[feature_cols]
    y = df["Trend"]
    return X, y