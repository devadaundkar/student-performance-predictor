import numpy as np
import pandas as pd

from preprocessing import FEATURE_COLUMNS, clean_data, engineer_features, get_features_and_target


def make_df():
    return pd.DataFrame({
        "study_hours": [4.0, 4.0, -1.0, 30.0, np.nan, 6.0],
        "attendance": [80.0, 80.0, 90.0, 90.0, 70.0, 130.0],
        "previous_score": [60.0, 60.0, 70.0, 70.0, 65.0, 75.0],
        "sleep_hours": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
        "extra_classes": [1, 1, 0, 0, 1, 0],
        "parental_education_years": [12, 12, 14, 14, 10, 16],
        "final_score": [70.0, 70.0, 65.0, 80.0, 60.0, 85.0],
    })


def test_clean_removes_duplicates_and_invalid_but_keeps_nan():
    cleaned = clean_data(make_df(), verbose=False)
    assert len(cleaned) == 2                       # 1 good row + 1 row with NaN study_hours
    assert cleaned["study_hours"].isna().sum() == 1  # NaN kept for the imputer
    assert cleaned["study_hours"].dropna().between(0, 16).all()
    assert cleaned["attendance"].between(0, 100).all()


def test_engineered_features_created_and_ordered():
    X, y = get_features_and_target(clean_data(make_df(), verbose=False))
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y)


def test_feature_values():
    row = pd.DataFrame([{"study_hours": 4.0, "attendance": 80.0, "previous_score": 60.0,
                         "sleep_hours": 6.5, "extra_classes": 1, "parental_education_years": 12}])
    out = engineer_features(row).iloc[0]
    assert np.isclose(out["sleep_deviation"], 1.0)
    assert np.isclose(out["effort_index"], 3.2)
    assert out["attendance_band"] == 1
    assert np.isclose(out["study_hours_log"], np.log1p(4.0))
