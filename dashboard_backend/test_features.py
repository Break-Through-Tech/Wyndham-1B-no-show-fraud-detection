
import pandas as pd
from pathlib import Path
from feature_engineering import build_features

ROOT = Path(__file__).resolve().parent.parent

# Load the same cleaned May data used in training
data = pd.read_parquet(
    ROOT / "data" / "revision4_combined_cleaned.parquet"
)
may = data[data["month"] == "May"].copy()

# Generate features using our new dashboard function
new_features = build_features(may)

# Load original modeling features
original = pd.read_parquet(
    ROOT / "data" / "may_modeling_data.parquet"
)

# Compare only Model 2's required features
model_features = [
    "booking_lead_days",
    "stay_length",
    "same_day_booking",
    "rate_code",
    "booking_channel",
    "member_tenure_days",
    "total_reservations",
    "single_night_ratio",
    "same_day_booking_count",
    "qns_count",
    "qxy_count",
    "qns_qxy_rate",
    "qns_qxy_points_earned",
    "qualification_code",
    "current_qns",
    "current_qxy",
    "points_earned",
    "room_revenue",
]

# Align records by confirmation number
new_features = new_features.set_index("confirmation_number")
original = original.set_index("confirmation_number")

new_features = new_features.sort_index()
original = original.sort_index()

print("New rows:", len(new_features))
print("Original rows:", len(original))

print("\nFeature differences:")
for col in model_features:
    a = new_features[col]
    b = original[col]

    matches = a.eq(b) | (a.isna() & b.isna())
    differences = (~matches).sum()

    print(f"{col}: {differences} differences")

print("\nFeature comparison complete!")
