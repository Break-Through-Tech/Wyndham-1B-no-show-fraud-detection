
import pandas as pd


def build_features(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Generate the existing Wyndham modeling features
    from combined reservation and stay data.
    """

    required = [
        "confirmation_number",
        "member_number",
        "booking_timestamp",
        "check_in_date",
        "check_out_date",
        "member_enrollment_date",
        "rate_code",
        "booking_channel",
        "qualification_code",
        "points_earned",
        "points_redeemed",
        "room_revenue",
    ]

    missing = [col for col in required if col not in combined.columns]

    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    if (
        combined["confirmation_number"].isna().any()
        or combined["confirmation_number"].duplicated().any()
    ):
        raise ValueError(
            "Confirmation numbers must be non-null and unique."
        )

    data = combined.copy()

    # Convert dates
    data["booking_timestamp"] = pd.to_datetime(
        data["booking_timestamp"]
    )
    data["check_in_date"] = pd.to_datetime(
        data["check_in_date"]
    )
    data["check_out_date"] = pd.to_datetime(
        data["check_out_date"]
    )
    data["member_enrollment_date"] = pd.to_datetime(
        data["member_enrollment_date"]
    )

    # Booking-time features
    data["booking_lead_days"] = (
        data["check_in_date"]
        - data["booking_timestamp"]
        .dt.tz_localize(None)
        .dt.normalize()
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

    # Sort reservations by booking time
    data = data.sort_values("booking_timestamp").copy()

    data["single_night"] = (
        data["stay_length"] == 1
    ).astype(int)

    # Historical booking behavior
    # Group simultaneous bookings to avoid counting
    # them as previous activity for one another.
    booking_history = (
        data[data["member_number"].notna()]
        .groupby(["member_number", "booking_timestamp"])
        .agg(
            bookings_at_time=("confirmation_number", "size"),
            single_nights_at_time=("single_night", "sum"),
            same_day_at_time=("same_day_booking", "sum"),
        )
        .reset_index()
        .sort_values(["member_number", "booking_timestamp"])
    )

    for col in [
        "bookings_at_time",
        "single_nights_at_time",
        "same_day_at_time",
    ]:
        booking_history["previous_" + col] = (
            booking_history.groupby("member_number")[col].cumsum()
            - booking_history[col]
        )

    booking_history["single_night_ratio"] = (
        booking_history["previous_single_nights_at_time"]
        / booking_history["previous_bookings_at_time"].replace(0, 1)
    )

    booking_history = booking_history.rename(
        columns={
            "previous_bookings_at_time": "total_reservations",
            "previous_same_day_at_time": "same_day_booking_count",
        }
    )

    data = data.merge(
        booking_history[
            [
                "member_number",
                "booking_timestamp",
                "total_reservations",
                "single_night_ratio",
                "same_day_booking_count",
            ]
        ],
        on=["member_number", "booking_timestamp"],
        how="left",
    )

    # Historical qualification and redemption behavior
    data["is_qns"] = (
        data["qualification_code"] == "QNS"
    ).astype(int)

    data["is_qxy"] = (
        data["qualification_code"] == "QXY"
    ).astype(int)

    data["qns_qxy_points"] = data["points_earned"].where(
        data["qualification_code"].isin(["QNS", "QXY"]),
        0,
    )

    data["was_redemption"] = (
        data["points_redeemed"].fillna(0) != 0
    ).astype(int)

    data["redeemed_points"] = (
        data["points_redeemed"].fillna(0).abs()
    )

    # Build historical member activity
    history = data[
        data["member_number"].notna()
    ].copy()

    # Approximation: stay information becomes
    # available at checkout.
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
            "total_points_redeemed",
        ]
    ].copy()

    # Attach only history before each booking
    members = data[
        data["member_number"].notna()
    ].copy()

    nonmembers = data[
        data["member_number"].isna()
    ].copy()

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
        allow_exact_matches=False,
    )

    data = pd.concat(
        [members, nonmembers],
        ignore_index=True,
    )

    history_features = [
        "qns_count",
        "qxy_count",
        "qns_qxy_rate",
        "qns_qxy_points_earned",
        "redemption_count",
        "total_points_redeemed",
    ]

    data[history_features] = (
        data[history_features].fillna(0)
    )

    # Model 2: current stay outcome
    data["current_qns"] = (
        data["qualification_code"] == "QNS"
    ).astype(int)

    data["current_qxy"] = (
        data["qualification_code"] == "QXY"
    ).astype(int)

    # Return features and identifiers
    features = [
        "confirmation_number",
        "member_number",
        "booking_timestamp",
        "history_date",
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
        "qualification_code",
        "current_qns",
        "current_qxy",
        "points_earned",
        "room_revenue",
    ]

    return data[features].copy()
