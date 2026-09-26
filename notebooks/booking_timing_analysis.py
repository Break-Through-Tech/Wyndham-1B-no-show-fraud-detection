import pandas as pd

data = pd.read_parquet("data/revision4_combined_cleaned.parquet")

data = data[data["member_number"].notna()].copy()

#all the dates
data["booking_timestamp"] = pd.to_datetime(data["booking_timestamp"])
data["check_in_date"] = pd.to_datetime(data["check_in_date"])
data["check_out_date"] = pd.to_datetime(data["check_out_date"])
data["cancellation_date"] = pd.to_datetime(data["cancellation_date"])
data["member_enrollment_date"] = pd.to_datetime(data["member_enrollment_date"])

#normalziing booking days cz this var becomes -1 
data["booking_lead_days"] = (
    data["check_in_date"] -
    data["booking_timestamp"].dt.tz_localize(None).dt.normalize()
).dt.days

data["stay_length"] = (
    data["check_out_date"] - data["check_in_date"]
).dt.days

data["single_night"] = (data["stay_length"] == 1).astype(int)
data["same_day_booking"] = (data["booking_lead_days"] == 0).astype(int)

# cancellation timing
data["cancel_days_before_checkin"] = (
    data["check_in_date"] - data["cancellation_date"]
).dt.days

data["late_cancel"] = (
    (data["qualification_code"] == "QXY")
).astype(int)

# member tenure at booking
data["member_tenure_days"] = (
    data["booking_timestamp"].dt.tz_localize(None)
    - data["member_enrollment_date"]
).dt.days

member_timing = data.groupby("member_number").agg(
    avg_booking_lead_days=("booking_lead_days", "mean"),
    avg_stay_length=("stay_length", "mean"),
    single_night_ratio=("single_night", "mean"),
    same_day_booking_count=("same_day_booking", "sum"),
    late_cancellation_rate=("late_cancel", "mean"),
    account_member_tenure=("member_tenure_days", "max")
).reset_index()

member_timing.to_parquet(
    "data/member_booking_timing_analysis.parquet",
    index=False
)

print(member_timing.head())
print("Members:", len(member_timing))

negative_lead = data[data["booking_lead_days"] < 0]

print("\nNegative booking lead rows:", len(negative_lead))

print(
    negative_lead[
        ["booking_timestamp", "check_in_date", "booking_lead_days"]
    ].head(10)
)