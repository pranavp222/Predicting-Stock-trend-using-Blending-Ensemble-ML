"""
strategy.py
-----------
Trading strategy backtest: applies the ensemble classifier's predictions
to generate a long-only strategy and evaluates performance with QuantStats.
"""

import matplotlib.pyplot as plt
import pandas as pd
import quantstats as qs


def compute_strategy_returns(df: pd.DataFrame, model, X) -> pd.Series:
    """
    Compute next-day strategy returns by multiplying shifted daily returns
    by the model's predictions (signal: 1 = long, 0 = flat).

    Parameters
    ----------
    df    : DataFrame containing the 'Daily_Returns' column.
    model : Fitted classifier with a predict() method.
    X     : Feature matrix aligned with *df*.

    Returns
    -------
    pd.Series of strategy returns.
    """
    df = df.copy()
    df["Strategy_Returns"] = df["Daily_Returns"].shift(-1) * model.predict(X)
    return df["Strategy_Returns"]


def plot_strategy_histogram(strategy_returns: pd.Series, train_size: int = 870) -> None:
    """Plot a histogram of out-of-sample strategy returns."""
    oos_returns = strategy_returns.iloc[train_size:]
    oos_returns.hist(figsize=(4, 4))
    plt.title("Strategy Returns Histogram")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def print_performance_report(strategy_returns: pd.Series, train_size: int = 870) -> None:
    """Print a full QuantStats performance report for out-of-sample returns."""
    qs.extend_pandas()
    oos_returns = strategy_returns.iloc[train_size:]
    print(qs.reports.full(oos_returns, figsize=(6, 6)))