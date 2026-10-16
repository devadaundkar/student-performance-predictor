"""
evaluate.py
------------
Metrics and diagnostic plots for regression models.

- MAE  (Mean Absolute Error): average miss, in score points. Easy to explain.
- RMSE (Root Mean Squared Error): like MAE but punishes big misses more.
- R^2: share of the score variation the model explains (1 = perfect,
  0 = no better than always guessing the average).
"""

import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test, verbose: bool = True) -> dict:
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)
    if verbose:
        print(f"  MAE  (avg error in points): {mae:.2f}")
        print(f"  RMSE (penalizes big errors): {rmse:.2f}")
        print(f"  R^2                        : {r2:.3f}")
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}


def plot_residuals(model, X_test, y_test, path: str):
    """Residual = actual - predicted. A good model's residuals scatter evenly
    around 0 with no pattern; a visible curve means the model misses something."""
    pred = model.predict(X_test)
    residuals = y_test - pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(pred, residuals, alpha=0.6)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted score")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residuals vs predictions")
    axes[1].scatter(y_test, pred, alpha=0.6)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    axes[1].plot(lims, lims, color="red", linestyle="--")
    axes[1].set_xlabel("Actual score")
    axes[1].set_ylabel("Predicted score")
    axes[1].set_title("Actual vs predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_feature_importance(model, X_test, y_test, path: str):
    """Permutation importance: shuffle one feature at a time and see how much
    the error increases. Works for ANY model (linear or tree-based)."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42,
        scoring="neg_root_mean_squared_error",
    )
    order = np.argsort(result.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(np.array(X_test.columns)[order], result.importances_mean[order],
            xerr=result.importances_std[order])
    ax.set_xlabel("Increase in RMSE when feature is shuffled")
    ax.set_title("Feature importance (permutation)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return dict(zip(X_test.columns, result.importances_mean.round(3).tolist()))
