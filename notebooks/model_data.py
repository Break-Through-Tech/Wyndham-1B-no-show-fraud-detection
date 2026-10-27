import pandas as pd

# Load cleaned reservation data and use May for initial model development
# Load known May fraud events and match them by confirmation number
# Create binary target: 1 = fraud, 0 = not labeled fraud

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


#Converting date columns to datetime so we can calculate time differences
# between booking, check in, check-out, and member enrollment dates

data["booking_timestamp"] = pd.to_datetime(data["booking_timestamp"])
data["check_in_date"] = pd.to_datetime(data["check_in_date"])
data["check_out_date"] = pd.to_datetime(data["check_out_date"])
data["member_enrollment_date"] = pd.to_datetime(
    data["member_enrollment_date"]
)

# MODEL 1: information available when reservation is booked
#removing timezone information and normalize the booking time to midnight
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

# put reservations in booking order
data = data.sort_values("booking_timestamp").copy()
#checking members with multiple bookings at the exact same time
#Simultaneous bookings should not count as previous activity for each other
same_time_bookings = data[
    data["member_number"].notna()
].duplicated(
    subset=["member_number", "booking_timestamp"],
    keep=False
)

print(
    "Reservations sharing a member and booking timestamp:",
    same_time_bookings.sum()
)

# previous reservation activity
data["single_night"] = (data["stay_length"] == 1).astype(int)

#Group reservations by member and booking timestamp
#Counting bookings, one-night stays, and same-day bookings made at each timestamp
#Grouping simultaneous bookings prevents them from counting as history for each other
booking_history = (
    data[data["member_number"].notna()]
    .groupby(["member_number", "booking_timestamp"])
    .agg(
        bookings_at_time=("confirmation_number", "size"),
        single_nights_at_time=("single_night", "sum"),
        same_day_at_time=("same_day_booking", "sum")
    )
    .reset_index()
    .sort_values(["member_number", "booking_timestamp"])
)

for col in ["bookings_at_time", "single_nights_at_time", "same_day_at_time"]:
    booking_history["previous_" + col] = (
        booking_history.groupby("member_number")[col].cumsum()
        - booking_history[col]
    )

booking_history["single_night_ratio"] = (
    booking_history["previous_single_nights_at_time"]
    / booking_history["previous_bookings_at_time"].replace(0, 1)
)

booking_history = booking_history.rename(columns={
    "previous_bookings_at_time": "total_reservations",
    "previous_same_day_at_time": "same_day_booking_count"
})

data = data.merge(
    booking_history[
        [
            "member_number",
            "booking_timestamp",
            "total_reservations",
            "single_night_ratio",
            "same_day_booking_count"
        ]
    ],
    on=["member_number", "booking_timestamp"],
    how="left"
)

# historical QNS, QXY, points, and redemption behavior
data["is_qns"] = (
    data["qualification_code"] == "QNS"
).astype(int)

data["is_qxy"] = (
    data["qualification_code"] == "QXY"
).astype(int)

data["qns_qxy_points"] = data["points_earned"].where(
    data["qualification_code"].isin(["QNS", "QXY"]),
    0
)

data["was_redemption"] = (
    data["points_redeemed"].fillna(0) != 0
).astype(int)

data["redeemed_points"] = (
    data["points_redeemed"].fillna(0).abs()
)

# build member history based on completed reservations
history = data[
    data["member_number"].notna()
].copy()

#here im approximating the historical stay info becomes avaialble at checkout date
#its an approximate, not verified posting date(can be changed according to wyndham data avalability)
history["history_date"] = history["check_out_date"]

history = history.sort_values(
    ["member_number", "history_date"]
)

history["qns_count"] = (
    history.groupby("member_number")["is_qns"].cumsum()
)

history["qxy_count"] = (
    history.groupby("member_number")["is_qxy"].cumsum()
)

history["completed_reservations"] = (
    history.groupby("member_number").cumcount() + 1
)

history["qns_qxy_points_earned"] = (
    history.groupby("member_number")["qns_qxy_points"].cumsum()
)

history["redemption_count"] = (
    history.groupby("member_number")["was_redemption"].cumsum()
)

history["total_points_redeemed"] = (
    history.groupby("member_number")["redeemed_points"].cumsum()
)

history["qns_qxy_rate"] = (
    (history["qns_count"] + history["qxy_count"])
    / history["completed_reservations"]
)

history = history[
    [
        "member_number",
        "history_date",
        "qns_count",
        "qxy_count",
        "qns_qxy_rate",
        "qns_qxy_points_earned",
        "redemption_count",
        "total_points_redeemed"
    ]
].copy()

# attach only member history known before each booking
members = data[
    data["member_number"].notna()
].copy()

nonmembers = data[
    data["member_number"].isna()
].copy()

# booking timestamp has UTC but check-out date does not
members["booking_timestamp"] = (
    members["booking_timestamp"].dt.tz_localize(None)
)

members = members.sort_values(
    ["booking_timestamp", "member_number"]
)

history = history.sort_values(
    ["history_date", "member_number"]
)

members = pd.merge_asof(
    members,
    history,
    left_on="booking_timestamp",
    right_on="history_date",
    by="member_number",
    direction="backward",
    allow_exact_matches=False
)

data = pd.concat(
    [members, nonmembers],
    ignore_index=True
)

history_features = [
    "qns_count",
    "qxy_count",
    "qns_qxy_rate",
    "qns_qxy_points_earned",
    "redemption_count",
    "total_points_redeemed"
]

data[history_features] = (
    data[history_features].fillna(0)
)

# MODEL 2: information available after outcome but before redemption
data["current_qns"] = (
    data["qualification_code"] == "QNS"
).astype(int)

data["current_qxy"] = (
    data["qualification_code"] == "QXY"
).astype(int)

# columns needed for both models
features = [
    "confirmation_number",
    "member_number",
    "booking_timestamp",  #Added for history timing validation
    "history_date",

    # model 1 - booking-time features
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
    "redemption_count",
    "total_points_redeemed",

    # model 2 - additional pre-redemption features
    "qualification_code",
    "current_qns",
    "current_qxy",
    "points_earned",
    "room_revenue",

    # target
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

invalid_history = (
    modeling_data["history_date"].notna()
    & (
        modeling_data["history_date"]
        >= modeling_data["booking_timestamp"]
    )
)

print("Future history violations:", invalid_history.sum())