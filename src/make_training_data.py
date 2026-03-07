# src/make_training_data.py
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Map task_type -> which survey column should be used as the baseline hours
TYPE_TO_SURVEY_TARGET = {
    "reading": "reading_hours",
    "essay": "essay_presentation_hours",
    "weekly_short": "weekly_short_hours",
    "weekly_long": "weekly_long_hours",
    "project": "project_hours",
    "midterm_easy": "easy_midterm_hours",
    "midterm_hard": "hard_midterm_hours",
    "final_easy": "easy_final_hours",
    "final_hard": "hard_final_hours",
    "capstone": "capstone_hours",
    "other": "reading_hours",  # fallback
}

def parse_date_safe(s):
    try:
        return datetime.fromisoformat(str(s)[:10]).date()
    except:
        return None

def build_rows(students: pd.DataFrame, tasks: pd.DataFrame, n_pairs: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # sample student-task pairs
    student_idx = rng.integers(0, len(students), size=n_pairs)
    task_idx = rng.integers(0, len(tasks), size=n_pairs)

    S = students.iloc[student_idx].reset_index(drop=True)
    T = tasks.iloc[task_idx].reset_index(drop=True)

    today = datetime.now().date()

    due_dates = T["due_date"].apply(parse_date_safe)
    days_until_due = due_dates.apply(lambda d: max(0, (d - today).days) if d else 30).astype(int)

    weight = pd.to_numeric(T.get("weight", 0), errors="coerce").fillna(0).astype(float).to_numpy()

    # baseline from survey for that task_type
    base = np.zeros(n_pairs, dtype=float)
    for i, tt in enumerate(T["task_type"].astype(str).tolist()):
        col = TYPE_TO_SURVEY_TARGET.get(tt, "reading_hours")
        base[i] = float(S.loc[i, col])

    # adjustments (simple but reasonable)
    urgency = np.exp(-days_until_due.to_numpy() / 14.0)          # closer due => larger
    weight_factor = 1.0 + (weight / 100.0) * 1.8                 # higher weight => more hours
    noise = rng.normal(0, 0.10, size=n_pairs)                    # 10% multiplicative noise

    y = base * weight_factor * (1.0 + 0.35 * urgency) * (1.0 + noise)
    y = np.clip(y, 0, None)

    # cap: a single task shouldn't exceed 60% of heavy week max
    cap = 0.60 * S["heavy_week_max"].to_numpy()
    y = np.minimum(y, cap)

    out = pd.DataFrame({
        "student_id": S["student_id"].to_numpy(),
        "task_id": T["task_id"].to_numpy(),
        "light_week_max": S["light_week_max"].to_numpy(),
        "heavy_week_max": S["heavy_week_max"].to_numpy(),
        "heavy_light_ratio": S["heavy_light_ratio"].to_numpy(),
        "task_type": T["task_type"].astype(str).to_numpy(),
        "weight": weight,
        "days_until_due": days_until_due.to_numpy(),
        "y_hours": y,
    })
    return out

def main():
    students_path = Path("data/synthetic/student_profiles_synth.csv")
    tasks_path = Path("data/raw/syllabus_tasks.csv")
    out_path = Path("data/synthetic/training_synth.csv")

    students = pd.read_csv(students_path)
    tasks = pd.read_csv(tasks_path)

    if len(tasks) == 0:
        raise ValueError("syllabus_tasks.csv has 0 rows. Task 2 didn't export items correctly.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    train = build_rows(students, tasks, n_pairs=80000, seed=42)
    train.to_csv(out_path, index=False)

    print(f"Saved: {out_path} rows={len(train)}")
    print(train.head(5))

if __name__ == "__main__":
    main()