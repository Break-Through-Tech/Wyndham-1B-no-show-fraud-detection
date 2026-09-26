import pandas as pd
pd.set_option("display.max_columns", None)

data = pd.read_parquet(
    "data/member_booking_timing_analysis.parquet"
)

fraud_labels = pd.read_csv(
    "data/revision4/member_ground_truth_may2026.csv"
)

data = data.merge(
    fraud_labels,
    on="member_number",
    how="left"
)

features = [
    "avg_booking_lead_days",
    "avg_stay_length",
    "single_night_ratio",
    "same_day_booking_count",
    "late_cancellation_rate",
    "account_member_tenure"
]

print("\nFRAUD VS LEGIT - BOOKING + TIMING")

print(
    data.groupby("is_fraud_account")[features].mean()
)