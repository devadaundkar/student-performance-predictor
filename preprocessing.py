"""
preprocessing.py
-----------------
Data cleaning and feature engineering (the Pandas + NumPy part).

Key design decisions (good interview talking points):
1. clean_data() only does things that are safe BEFORE the train/test split
   (dropping duplicates, removing impossible rows). It does NOT fill missing
   values, because computing a mean/median on the full dataset would leak
   information from the test set into training. Imputation happens later,
   inside the scikit-learn Pipeline, fitted on training data only.
2. engineer_features() is ONE shared function used by both training and the
   Streamlit app, so the model always sees identical features at prediction
   time (avoids "training-serving skew").
"""

import numpy as np
import pandas as pd

RAW_FEATURES = [
    "study_hours", "attendance", "previous_score",
    "sleep_hours", "extra_classes", "parental_education_years",
]
ENGINEERED_FEATURES = [
    "study_hours_log", "sleep_deviation", "effort_index", "attendance_band",
]
FEATURE_COLUMNS = RAW_FEATURES + ENGINEERED_FEATURES
TARGET = "final_score"

# Physically possible ranges; anything outside is a data-entry error
VALID_RANGES = {
    "study_hours": (0, 16),
    "attendance": (0, 100),
    "previous_score": (0, 100),
    "sleep_hours": (0, 16),
}


def load_data(path: str = "data/students.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop duplicates, rows without a target, and rows with impossible values.
    Missing FEATURE values are kept as NaN and imputed later in the Pipeline."""
    df = df.copy()
    start = len(df)

    df = df.drop_duplicates()
    n_dupes = start - len(df)

    before = len(df)
    df = df.dropna(subset=[TARGET])          # can't learn from an unknown answer
    n_no_target = before - len(df)

    before = len(df)
    for col, (low, high) in VALID_RANGES.items():
        in_range = df[col].between(low, high) | df[col].isna()   # keep NaN for the imputer
        df = df[in_range]
    n_invalid = before - len(df)

    if verbose:
        print(f"Cleaning: {start} rows in -> dropped {n_dupes} duplicates, "
              f"{n_no_target} without target, {n_invalid} invalid -> {len(df)} rows left")
        print(f"Missing feature values still to impute: {int(df[RAW_FEATURES].isna().sum().sum())}")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new, more informative features from the raw ones."""
    df = df.copy()

    # Studying has diminishing returns -> log compresses large values
    df["study_hours_log"] = np.log1p(df["study_hours"].clip(lower=0))

    # Distance from a healthy ~7.5 h of sleep (too little OR too much is bad)
    df["sleep_deviation"] = (df["sleep_hours"] - 7.5).abs()

    # Interaction: studying a lot AND attending class beats either alone
    df["effort_index"] = df["study_hours"] * df["attendance"] / 100

    # Bucket attendance: 0 = low (<=65), 1 = medium (65-85), 2 = high (>85)
    df["attendance_band"] = pd.cut(
        df["attendance"], bins=[-np.inf, 65, 85, np.inf], labels=False
    )
    return df


def get_features_and_target(df: pd.DataFrame):
    """Return X (features, including engineered ones) and y (final_score)."""
    df = engineer_features(df)
    return df[FEATURE_COLUMNS], df[TARGET]


if __name__ == "__main__":
    raw = load_data()
    print("Raw shape:", raw.shape)
    print(raw.isna().sum())
    clean = clean_data(raw)
    X, y = get_features_and_target(clean)
    print("\nFeature columns:", list(X.columns))
    print(X.head())
