import pandas as pd

data = pd.read_parquet("data/revision4_combined_cleaned.parquet")

months = ["may", "june", "july"]

fraud_events = []

for month in months:
    events = pd.read_csv(
        f"data/revision4/fraud_event_ground_truth_{month}2026.csv"
    )
    fraud_events.append(events)

fraud_events = pd.concat(fraud_events, ignore_index=True)

#getting fraud event rows
fraud_data = data.merge(
    fraud_events,
    on=["confirmation_number", "member_number"],
    how="inner"
)

print("Fraud events:", len(fraud_data))

print("\nFraud mechanisms:")
print(fraud_data["fraud_mechanism"].value_counts())

print("\nQualification codes:")
print(fraud_data["qualification_code"].value_counts())

print("\nRate codes:")
print(fraud_data["rate_code"].value_counts())

print("\nBooking channels:")
print(fraud_data["booking_channel"].value_counts())

print("\nPoints earned:")
print(fraud_data["points_earned"].describe())

print("\nRoom revenue:")
print(fraud_data["room_revenue"].describe())

#timing
fraud_data["booking_timestamp"] = pd.to_datetime(
    fraud_data["booking_timestamp"]
)

fraud_data["check_in_date"] = pd.to_datetime(
    fraud_data["check_in_date"]
)

fraud_data["booking_lead_days"] = (
    fraud_data["check_in_date"]
    - fraud_data["booking_timestamp"].dt.tz_localize(None).dt.normalize()
).dt.days

fraud_data["stay_length"] = (
    pd.to_datetime(fraud_data["check_out_date"])
    - fraud_data["check_in_date"]
).dt.days

print("\nBooking lead days:")
print(fraud_data["booking_lead_days"].describe())

print("\nStay length:")
print(fraud_data["stay_length"].describe())

print("\nFraud behavior by mechanism:")
print(
    fraud_data.groupby("fraud_mechanism")[
        ["points_earned", "room_revenue", "booking_lead_days", "stay_length"]
    ].mean()
)