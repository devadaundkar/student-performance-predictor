"""
train_model.py
----------------
Trains several regression models, picks the best using cross-validation,
evaluates it on a held-out test set, and saves it for the Streamlit app.

Pipeline for every model:  impute missing values -> (scale) -> model
Putting these steps inside a scikit-learn Pipeline means they are fitted on
TRAINING data only (no data leakage) and travel together with the model.

Model selection uses cross-validation on the training set (NOT the test set),
so the test set stays an honest, untouched final exam.
"""

import json
import os

import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import evaluate_model, plot_feature_importance, plot_residuals
from preprocessing import FEATURE_COLUMNS, clean_data, get_features_and_target, load_data

MODEL_PATH = "model.joblib"
REPORT_DIR = "reports"
RANDOM_STATE = 42


def make_pipeline(estimator, scale: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:                       # scaling matters for linear models, not for trees
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def train_and_save_model(model_path: str = MODEL_PATH, data_path: str = "data/students.csv"):
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 1. Load, clean, engineer features
    df = clean_data(load_data(data_path))
    X, y = get_features_and_target(df)

    # 2. Hold out 20% as a final test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}\n")

    # 3. Candidate models
    candidates = {
        "Linear Regression": make_pipeline(LinearRegression(), scale=True),
        "Ridge": make_pipeline(Ridge(alpha=1.0), scale=True),
        "Random Forest": make_pipeline(
            RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)),
        "Gradient Boosting": make_pipeline(
            GradientBoostingRegressor(random_state=RANDOM_STATE)),
    }

    # 3b. Hyperparameter tuning for Gradient Boosting (grid search + 5-fold CV)
    grid = GridSearchCV(
        make_pipeline(GradientBoostingRegressor(random_state=RANDOM_STATE)),
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [2, 3],
            "model__learning_rate": [0.05, 0.1],
        },
        cv=5, scoring="neg_root_mean_squared_error",
    )
    grid.fit(X_train, y_train)
    candidates["Gradient Boosting (tuned)"] = grid.best_estimator_
    print(f"Best Gradient Boosting params: {grid.best_params_}\n")

    # 4. Cross-validate and evaluate every candidate
    results = []
    for name, pipe in candidates.items():
        cv_rmse = -cross_val_score(
            pipe, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error"
        ).mean()
        pipe.fit(X_train, y_train)
        print(f"=== {name} ===  (5-fold CV RMSE: {cv_rmse:.2f})")
        test_metrics = evaluate_model(pipe, X_test, y_test)
        results.append({"model": name, "cv_rmse": round(float(cv_rmse), 3),
                        **{k: round(v, 3) for k, v in test_metrics.items()}})
        print()

    # 5. Choose the winner by CROSS-VALIDATION error (not by test score)
    best = min(results, key=lambda r: r["cv_rmse"])
    best_model = candidates[best["model"]]
    print(f"Selected model: {best['model']} (lowest CV RMSE = {best['cv_rmse']})")

    # 6. Diagnostics for the chosen model
    plot_residuals(best_model, X_test, y_test, f"{REPORT_DIR}/residuals.png")
    importance = plot_feature_importance(
        best_model, X_test, y_test, f"{REPORT_DIR}/feature_importance.png")

    # 7. Save the model together with everything the app needs
    summary = {
        "model_name": best["model"],
        "test_metrics": {k: best[k] for k in ("mae", "rmse", "r2")},
        "comparison": results,
        "feature_importance": importance,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_columns": FEATURE_COLUMNS,
    }
    joblib.dump({"model": best_model, **summary}, model_path)
    with open(f"{REPORT_DIR}/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved model -> {model_path} and report -> {REPORT_DIR}/")
    return best_model, summary


if __name__ == "__main__":
    train_and_save_model()
