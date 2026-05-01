"""
main.py
-------
End-to-end pipeline: data → EDA → model training → evaluation → backtest.
Run with:  python main.py
"""

import warnings

from sklearn.model_selection import train_test_split

from data import build_features, download_data, fix_multicollinearity, get_feature_target
from eda import run_eda
from models import build_stacking_ensemble, evaluate_model, plot_roc_curve, tune_base_models
from strategy import compute_strategy_returns, plot_strategy_histogram, print_performance_report

warnings.filterwarnings("ignore", category=UserWarning)

TICKER = "TSLA"
START = "2019-01-05"
END = "2024-01-05"
TRAIN_SIZE = 870  # approximate training-set boundary for OOS strategy eval


def main():
    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("Downloading data ...")
    raw = download_data(TICKER, START, END)

    print("Building features ...")
    df = build_features(raw)

    # ------------------------------------------------------------------
    # 2. EDA
    # ------------------------------------------------------------------
    run_eda(df)

    # Fix multicollinearity identified during EDA
    df = fix_multicollinearity(df)

    # ------------------------------------------------------------------
    # 3. Train / val / test split
    # ------------------------------------------------------------------
    X, y = get_feature_target(df)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # ------------------------------------------------------------------
    # 4. Base models (no tuning) — quick sanity check
    # ------------------------------------------------------------------
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    for name, clf in [
        ("Decision Tree Classifier", DecisionTreeClassifier()),
        ("Random Forest Classifier", RandomForestClassifier()),
        ("KNN Classifier", KNeighborsClassifier()),
        ("XGBoost Classifier", XGBClassifier()),
    ]:
        clf.fit(X_train, y_train)
        evaluate_model(clf, X_val, y_val, X_test, y_test, model_name=name)

    # ------------------------------------------------------------------
    # 5. Hyperparameter tuning
    # ------------------------------------------------------------------
    print("\nTuning base models (this may take a few minutes) ...")
    grid_searches = tune_base_models(X_train, y_train, cv=5)

    # ------------------------------------------------------------------
    # 6. Stacking ensemble
    # ------------------------------------------------------------------
    print("\nBuilding stacking ensemble ...")
    stacked = build_stacking_ensemble(grid_searches)
    stacked.fit(X_train, y_train)

    evaluate_model(stacked, X_val, y_val, X_test, y_test, model_name="Stacked Ensemble Classifier")
    plot_roc_curve(stacked, X_val, y_val)

    # ------------------------------------------------------------------
    # 7. Backtest
    # ------------------------------------------------------------------
    print("\nRunning backtest ...")
    strategy_returns = compute_strategy_returns(df, stacked, X)
    plot_strategy_histogram(strategy_returns, train_size=TRAIN_SIZE)
    print_performance_report(strategy_returns, train_size=TRAIN_SIZE)


if __name__ == "__main__":
    main()