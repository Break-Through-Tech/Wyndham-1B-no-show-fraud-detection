
from pathlib import Path

import joblib
import pandas as pd

from feature_engineering import build_features

ROOT = Path(__file__).resolve().parent.parent

# Load the saved Model 2 pipeline
model = joblib.load(ROOT / "models" / "model2_pipeline.joblib")


def predict_fraud(combined: pd.DataFrame) -> pd.DataFrame:
    # Generate the same features used during training
    prepared = build_features(combined)

    # Select Model 2's 18 input features in training order
    feature_columns = model.feature_names_in_.tolist()
    X = prepared[feature_columns]

    # Generate predictions and risk scores
    prepared["fraud_prediction"] = model.predict(X)
    prepared["fraud_risk_score"] = model.predict_proba(X)[:, 1]

    return prepared


if __name__ == "__main__":
    # Test using the cleaned May data
    data = pd.read_parquet(
        ROOT / "data" / "revision4_combined_cleaned.parquet"
    )

    may = data[data["month"] == "May"].copy()

    results = predict_fraud(may)

    print("Total reservations:", len(results))
    print("Flagged reservations:", results["fraud_prediction"].sum())

    print("\nSample flagged reservations:")
    print(
        results.loc[
            results["fraud_prediction"] == 1,
            [
                "confirmation_number",
                "member_number",
                "fraud_risk_score",
            ],
        ].head(10)
    )

    print("\nPrediction pipeline test successful!")
