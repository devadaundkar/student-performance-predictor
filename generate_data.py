"""
generate_data.py
-----------------
Generates a synthetic student performance dataset and saves it as a CSV.

IMPORTANT (be upfront about this in interviews): this data is SYNTHETIC.
The final score is built from a formula + random noise, so the results show
that the pipeline works, not how well it would predict real students.
To use real data, put a CSV with the same column names at data/students.csv.

Realistic touches built into the formula:
- Diminishing returns: the 8th hour of studying helps less than the 1st.
- Sleep has a sweet spot (~7.5 h); too little OR too much hurts.
- Random noise (std = 4 points) so the target is never perfectly predictable.

With messy=True we also inject missing values, duplicates and impossible
values, so the cleaning step in preprocessing.py has real work to do.
"""

import os
import numpy as np
import pandas as pd


def generate_student_data(n_students: int = 500, seed: int = 42, messy: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Academic features ---
    attendance = rng.uniform(50, 100, n_students)              # % attendance
    previous_score = rng.uniform(35, 100, n_students)          # last exam score
    parental_education_years = rng.integers(8, 20, n_students)  # years of schooling

    # --- Behavioural features ---
    study_hours = rng.uniform(0, 10, n_students)               # hours/day studying
    sleep_hours = rng.uniform(4, 9, n_students)                # hours/day sleep
    extra_classes = rng.integers(0, 2, n_students)             # 0 = No, 1 = Yes

    noise = rng.normal(0, 4, n_students)

    study_effect = 32 * (1 - np.exp(-study_hours / 4))         # diminishing returns
    sleep_effect = -0.8 * (sleep_hours - 7.5) ** 2             # best near 7.5 h

    final_score = (
        study_effect
        + attendance * 0.25
        + previous_score * 0.35
        + sleep_effect
        + extra_classes * 4
        + parental_education_years * 0.3
        + noise
    )
    final_score = np.clip(final_score, 0, 100)

    df = pd.DataFrame({
        "study_hours": study_hours.round(2),
        "attendance": attendance.round(2),
        "previous_score": previous_score.round(2),
        "sleep_hours": sleep_hours.round(2),
        "extra_classes": extra_classes,
        "parental_education_years": parental_education_years,
        "final_score": final_score.round(2),
    })

    return _inject_mess(df, seed) if messy else df


def _inject_mess(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Make the data look like real-world data: gaps, duplicates, bad values."""
    rng = np.random.default_rng(seed + 1)
    df = df.copy()
    n = len(df)

    # ~3% missing values in four feature columns
    for col in ["study_hours", "attendance", "previous_score", "sleep_hours"]:
        idx = rng.choice(n, size=int(0.03 * n), replace=False)
        df.loc[idx, col] = np.nan

    # a few impossible values (data-entry errors)
    bad = rng.choice(n, size=9, replace=False)
    df.loc[bad[:3], "study_hours"] = -2.0      # negative hours
    df.loc[bad[3:6], "study_hours"] = 30.0     # more hours than a day allows
    df.loc[bad[6:], "attendance"] = 130.0      # attendance above 100%

    # a few duplicated rows
    df = pd.concat([df, df.sample(5, random_state=seed)], ignore_index=True)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_student_data()
    df.to_csv("data/students.csv", index=False)
    print(f"Generated {len(df)} student records -> data/students.csv")
    print(df.head())
