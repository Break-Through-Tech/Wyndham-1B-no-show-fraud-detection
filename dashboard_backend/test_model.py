
from pathlib import Path

import joblib
import pandas as pd

#Locate files from the project root
ROOT = Path(__file__).resolve().parent.parent

#Loading the saved Model 2
model = joblib.load(ROOT / "models" / "model2_pipeline.joblib")

#Loading existing prepared May data
data = pd.read_parquet(ROOT / "data" / "may_modeling_data.parquet")

#Selecting the features the model was trained on
features = model.feature_names_in_.tolist()
X = data[features].head(10)

#generating predictions
predictions = model.predict(X)
risk_scores = model.predict_proba(X)[:, 1]

#dispalying results
results = pd.DataFrame({
    "prediction": predictions,
    "fraud_risk_score": risk_scores
})

print(results)
print("\nModel 2 prediction test successful!")
