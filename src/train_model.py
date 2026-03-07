# src/train_model.py
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

def main():
    data_path = Path("data/synthetic/training_synth.csv")
    model_path = Path("data/model_hours_xgb.joblib")

    df = pd.read_csv(data_path)

    y = df["y_hours"].astype(float)
    X = df.drop(columns=["y_hours"])

    cat_cols = ["task_type"]
    num_cols = ["light_week_max", "heavy_week_max", "heavy_light_ratio", "weight", "days_until_due"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        objective="reg:squarederror",
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    joblib.dump(pipe, model_path)
    print(f"Saved model to: {model_path}")

if __name__ == "__main__":
    main()