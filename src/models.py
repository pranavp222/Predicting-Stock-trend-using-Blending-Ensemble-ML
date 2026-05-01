"""
models.py
---------
Base model definitions, hyperparameter tuning, blending ensemble construction,
and evaluation utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

PARAM_GRIDS = {
    "decision_tree": {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "knn": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "xgboost": {
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "n_estimators": [50, 100, 200],
        "subsample": [0.8, 0.9, 1.0],
    },
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_val, y_val, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluate a trained classifier on validation and test sets.

    Prints ROC-AUC, confusion matrix, classification report, and accuracy,
    and displays the confusion matrix as a heatmap.

    Returns a dict of metrics.
    """
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_val, y_val_pred)
    conf_matrix = confusion_matrix(y_val, y_val_pred)
    class_report = classification_report(y_val, y_val_pred)
    accuracy = accuracy_score(y_test, y_test_pred) * 100.0

    print(f"\n===== {model_name} Evaluation =====")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"\nConfusion Matrix:\n{conf_matrix}")
    print(f"\nClassification Report:\n{class_report}")
    print(f"Accuracy (Test): {accuracy:.2f}%")

    plt.figure(figsize=(4, 4))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    plt.show()

    return {
        "model": model_name,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
    }


def plot_roc_curve(model, X_val, y_val) -> None:
    """Plot the ROC curve for a classifier that supports predict_proba."""
    proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, proba)
    auc = roc_auc_score(y_val, proba)

    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, color="navy", lw=2, label=f"ROC Curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Stacked Ensemble Classifier")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def tune_base_models(X_train, y_train, cv: int = 5) -> dict:
    """
    Run GridSearchCV for each base model and return the best estimators.

    Returns
    -------
    dict mapping model key -> fitted GridSearchCV object
    """
    base_estimators = {
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier(),
        "knn": KNeighborsClassifier(),
        "xgboost": XGBClassifier(),
    }

    searches = {}
    for name, estimator in base_estimators.items():
        print(f"Tuning {name} ...")
        gs = GridSearchCV(estimator, PARAM_GRIDS[name], cv=cv, n_jobs=-1)
        gs.fit(X_train, y_train)
        print(f"  Best params: {gs.best_params_}")
        searches[name] = gs

    return searches


def build_stacking_ensemble(grid_searches: dict) -> StackingClassifier:
    """
    Build a StackingClassifier from tuned base models with Logistic Regression
    as the final meta-estimator.
    """
    estimators = [
        (name, gs.best_estimator_)
        for name, gs in grid_searches.items()
    ]
    stacked = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
    )
    return stacked