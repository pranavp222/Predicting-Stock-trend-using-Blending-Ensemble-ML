"""
eda.py
------
Exploratory data analysis utilities: summary stats, correlation heatmap,
VIF computation, and regression scatter plots.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------------

def calc_vif(data: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with Variance Inflation Factors for each feature."""
    vif = pd.DataFrame()
    vif["Features"] = data.columns
    vif["VIF"] = [
        variance_inflation_factor(data.values, i)
        for i in range(data.shape[1])
    ]
    return vif


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(data: pd.DataFrame) -> None:
    """Plot a correlation heatmap for all columns in *data*."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(data.corr(), ax=ax, annot=True)
    plt.tight_layout()
    plt.show()


def plot_regression_scatter(df: pd.DataFrame) -> None:
    """Plot regression scatter plots for key feature pairs."""
    pairs = [
        ("Open-Close", "Daily_Returns"),
        ("Daily_Returns", "ret_5"),
        ("Past_Returns", "ret_5"),
    ]
    for x_col, y_col in pairs:
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.regplot(x=df[x_col], y=df[y_col], ax=ax)
        ax.set_title(f"{x_col} vs {y_col}")
        plt.tight_layout()
        plt.show()


def run_eda(df: pd.DataFrame) -> None:
    """Run full EDA: summary stats, correlation heatmap, VIF, scatter plots."""
    print("=== Summary Statistics ===")
    print(df.describe())

    print("\n=== Correlation Heatmap ===")
    plot_correlation_heatmap(df)

    print("\n=== VIF (all features) ===")
    print(calc_vif(df.iloc[:, :-1]))

    print("\n=== Regression Scatter Plots ===")
    plot_regression_scatter(df)