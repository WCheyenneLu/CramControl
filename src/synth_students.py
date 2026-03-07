# src/synth_students.py
import numpy as np
import pandas as pd
from pathlib import Path
from config import BOUNDS 


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def jitter_bootstrap(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    boot = df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    out = boot.copy()

    for col in df.columns:
        if col == "student_id":
            continue
        s = df[col].dropna()
        if len(s) < 2:
            continue

        std = float(s.std())
        noise = rng.normal(0, max(0.25, 0.15 * std), size=n)
        out[col] = out[col] + noise

        if col in BOUNDS:
            lo, hi = BOUNDS[col]
            out[col] = out[col].clip(lo, hi)

    out["heavy_light_ratio"] = (out["heavy_week_max"] + 1e-6) / (out["light_week_max"] + 1e-6)
    return out


def find_col(columns, must_contain_list):
    # returns the first column whose name contains all keywords (case-insensitive)
    for c in columns:
        c_lower = c.lower()
        if all(k.lower() in c_lower for k in must_contain_list):
            return c
    return None


def main():
    print("STARTING synth_students.py")

    raw_path = Path("data/raw/survey_responses.csv")
    out_path = Path("data/synthetic/student_profiles_synth.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_path)

    COLUMN_KEYWORDS = {
        "light_week_max": ["light week", "maximum", "hours"],
        "heavy_week_max": ["heavy week", "maximum", "hours"],
        "reading_hours": ["reading assignment"],
        "essay_presentation_hours": ["essay", "presentation"],
        "weekly_short_hours": ["weekly", "shorter"],
        "weekly_long_hours": ["weekly", "longer"],
        "project_hours": ["larger projects"],
        "easy_midterm_hours": ["easier midterm"],
        "hard_midterm_hours": ["harder midterm"],
        "easy_final_hours": ["easier final"],
        "hard_final_hours": ["harder final"],
        "capstone_hours": ["capstone"],
    }

    df = pd.DataFrame()

    for new_name, keys in COLUMN_KEYWORDS.items():
        col = find_col(raw.columns, keys)
        if col is None:
            raise ValueError(
                f"Could not find a column for '{new_name}'. "
                f"Looking for keywords: {keys}. "
                f"Available columns: {list(raw.columns)}"
            )
        df[new_name] = clean_numeric(raw[col])

    df = df.dropna()

    synth = jitter_bootstrap(df, n=5000, seed=42)
    synth.insert(0, "student_id", np.arange(len(synth)))

    synth.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  rows={len(synth)}")


if __name__ == "__main__":
    main()
