import pandas as pd

# only using May data to prevent leakage
data = pd.read_parquet("data/revision4_combined_cleaned.parquet")
data = data[data["month"] == "May"].copy()

# fraud event labels
fraud = pd.read_csv(
    "data/revision4/fraud_event_ground_truth_may2026.csv"
)

fraud_confirmations = set(fraud["confirmation_number"])

data["is_fraud_event"] = (
    data["confirmation_number"].isin(fraud_confirmations)
).astype(int)

# dates
data["booking_timestamp"] = pd.to_datetime(data["booking_timestamp"])
data["check_in_date"] = pd.to_datetime(data["check_in_date"])
data["check_out_date"] = pd.to_datetime(data["check_out_date"])
data["member_enrollment_date"] = pd.to_datetime(data["member_enrollment_date"])

# event features
data["booking_lead_days"] = (
    data["check_in_date"]
    - data["booking_timestamp"].dt.tz_localize(None).dt.normalize()
).dt.days

data["stay_length"] = (
    data["check_out_date"] - data["check_in_date"]
).dt.days

data["same_day_booking"] = (
    data["booking_lead_days"] == 0
).astype(int)

data["member_tenure_days"] = (
    data["booking_timestamp"].dt.tz_localize(None)
    - data["member_enrollment_date"]
).dt.days

# put reservations in time order
data = data.sort_values("booking_timestamp")

# previous reservation activity
data["total_reservations"] = (
    data.groupby("member_number").cumcount()
)

# previous single-night booking ratio
data["single_night"] = (
    data["stay_length"] == 1
).astype(int)

previous_single_nights = (
    data.groupby("member_number")["single_night"].cumsum()
    - data["single_night"]
)

data["single_night_ratio"] = (
    previous_single_nights
    / data["total_reservations"].replace(0, 1)
)

# previous same-day bookings
data["same_day_booking_count"] = (
    data.groupby("member_number")["same_day_booking"].cumsum()
    - data["same_day_booking"]
)

# modeling columns
features = [
    "confirmation_number",
    "member_number",
    "booking_lead_days",
    "stay_length",
    "same_day_booking",
    "rate_code",
    "booking_channel",
    "member_tenure_days",
    "total_reservations",
    "single_night_ratio",
    "same_day_booking_count",
    "is_fraud_event"
]

modeling_data = data[features].copy()

modeling_data.to_parquet(
    "data/may_modeling_data.parquet",
    index=False
)

print("Rows:", len(modeling_data))

print("\nFraud labels:")
print(modeling_data["is_fraud_event"].value_counts())

print("\nColumns:")
print(modeling_data.columns.tolist())

print("\nMissing values:")
print(modeling_data.isna().sum())

print("\nSaved data/may_modeling_data.parquet")