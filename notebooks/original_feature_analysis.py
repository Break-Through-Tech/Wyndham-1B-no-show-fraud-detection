
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================
# CONFIGURATION
# ==========================================

RAW_PATH = Path("data/revision4_combined_cleaned.parquet")
MAY_PATH = Path("data/may_modeling_data.parquet")
OUTPUT = Path("outputs/original_feature_analysis")

OUTPUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (11, 6),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 170
})

TARGET = "is_fraud_event"

RAW_COLUMNS = [
    "confirmation_number",
    "booking_timestamp",
    "check_in_date",
    "check_out_date",
    "cancellation_date",
    "rate_code",
    "room_revenue",
    "points_redeemed",
    "booking_channel",
    "site_id",
    "site_brand",
    "member_number",
    "member_enrollment_date",
    "member_enrollment_source",
    "points_earned",
    "qualification_code",
    "month"
]


# ==========================================
# HELPERS
# ==========================================

def save(name):
    plt.tight_layout()
    plt.savefig(OUTPUT / f"{name}.png", bbox_inches="tight")
    plt.close()
    print("Saved:", name)


def add_bar_labels(ax, suffix="", decimals=2):
    for bar in ax.patches:
        value = bar.get_height()
        ax.annotate(
            f"{value:.{decimals}f}{suffix}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
        )


def category_graph(data, column, filename, title, top_n=12):
    grouped = (
        data.groupby(column, dropna=False)[TARGET]
        .agg(["sum", "count"])
        .rename(columns={"sum": "fraud", "count": "total"})
    )

    grouped["fraud_rate"] = (
        grouped["fraud"] / grouped["total"] * 100
    )

    grouped = grouped.sort_values(
        "total", ascending=False
    ).head(top_n)

    grouped = grouped.sort_values("fraud_rate")

    labels = [
        "Missing" if pd.isna(x) else str(x)
        for x in grouped.index
    ]

    fig, ax = plt.subplots(figsize=(11, max(5, len(grouped) * 0.55)))

    bars = ax.barh(
        labels,
        grouped["fraud_rate"],
        color="#3977b8"
    )

    max_rate = grouped["fraud_rate"].max()
    ax.set_xlim(0, max(max_rate * 1.35, 0.05))

    for bar, (_, row) in zip(bars, grouped.iterrows()):
        ax.text(
            bar.get_width() + max(max_rate * 0.02, 0.002),
            bar.get_y() + bar.get_height() / 2,
            f'{row["fraud_rate"]:.3f}%  '
            f'({int(row["fraud"]):,}/{int(row["total"]):,})',
            va="center",
            fontsize=9
        )

    ax.set_xlabel("Fraud rate (%)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.15)

    save(filename)


def binned_graph(data, values, bins, labels, filename, title):
    subset = pd.DataFrame({
        "value": values,
        TARGET: data[TARGET]
    })

    subset["group"] = pd.cut(
        subset["value"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    grouped = subset.groupby(
        "group", observed=False
    )[TARGET].agg(["sum", "count"])

    grouped["fraud_rate"] = np.where(
        grouped["count"] > 0,
        grouped["sum"] / grouped["count"] * 100,
        0
    )

    fig, ax = plt.subplots()

    bars = ax.bar(
        grouped.index.astype(str),
        grouped["fraud_rate"],
        color="#3977b8"
    )

    for bar, (_, row) in zip(bars, grouped.iterrows()):
        ax.annotate(
            f'{row["fraud_rate"]:.2f}%\n'
            f'({int(row["sum"]):,}/{int(row["count"]):,})',
            (
                bar.get_x() + bar.get_width() / 2,
                bar.get_height()
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9
        )

    ax.set_ylim(
        0,
        max(grouped["fraud_rate"].max() * 1.3, 0.05)
    )

    ax.set_ylabel("Fraud rate (%)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.15)

    save(filename)


def numeric_distribution(data, column, filename, title,
                         bins, labels):
    binned_graph(
        data,
        pd.to_numeric(data[column], errors="coerce"),
        bins,
        labels,
        filename,
        title
    )


# ==========================================
# 1. LOAD ORIGINAL MAY DATA
# ==========================================

print("Loading original data...")

raw = pd.read_parquet(
    RAW_PATH,
    columns=RAW_COLUMNS,
    filters=[("month", "==", "May")]
)

# Month may be stored in a different representation.
# If no rows match, stop rather than silently
# producing empty or misleading graphs.
if raw.empty:
    raise ValueError(
        "No rows matched month == '2026-05'. "
        "Check how the month column is stored."
    )

print("Original May rows:", len(raw))

# The modeling file must contain the confirmation number
# to safely attach labels to the original reservations.
labels = pd.read_parquet(
    MAY_PATH,
    columns=["confirmation_number", TARGET]
)

if labels["confirmation_number"].duplicated().any():
    raise ValueError(
        "Duplicate confirmation numbers in May modeling labels. "
        "A reservation-level composite key is needed."
    )

if raw["confirmation_number"].duplicated().any():
    raise ValueError(
        "Duplicate confirmation numbers in original May data. "
        "A reservation-level composite key is needed."
    )

df = raw.merge(
    labels,
    on="confirmation_number",
    how="left",
    validate="one_to_one",
    indicator=True
)

if (df["_merge"] != "both").any():
    raise ValueError(
        "Some original reservations have no May fraud label. "
        "Check the May filter and matching keys."
    )

df = df.drop(columns="_merge")

print("Labeled rows:", len(df))
print("Fraud rows:", int(df[TARGET].sum()))


# ==========================================
# 2. CONVERT DATES
# ==========================================

date_columns = [
    "booking_timestamp",
    "check_in_date",
    "check_out_date",
    "member_enrollment_date",
    "cancellation_date"
]

for column in date_columns:
    df[column] = pd.to_datetime(
        df[column],
        errors="coerce",
        utc=True
    ).dt.tz_localize(None)


# ==========================================
# 3. ORIGINAL DATE FIELDS -> ENGINEERED FEATURES
# ==========================================

# Booking lead time
lead_days = (
    df["check_in_date"].dt.normalize()
    - df["booking_timestamp"].dt.normalize()
).dt.days

binned_graph(
    df,
    lead_days,
    [-np.inf, 0, 3, 7, 30, 90, np.inf],
    ["Same day", "1-3", "4-7", "8-30", "31-90", "91+"],
    "01_booking_dates_to_lead_time",
    "Original Booking Dates -> Booking Lead Time"
)

# Stay length
stay_length = (
    df["check_out_date"] - df["check_in_date"]
).dt.days

binned_graph(
    df,
    stay_length,
    [-np.inf, 1, 2, 3, 5, np.inf],
    ["1 night or less", "2", "3", "4-5", "6+"],
    "02_checkin_checkout_to_stay_length",
    "Original Stay Dates -> Length of Stay"
)

# Account age at booking
member_tenure = (
    df["booking_timestamp"].dt.normalize()
    - df["member_enrollment_date"].dt.normalize()
).dt.days

binned_graph(
    df,
    member_tenure,
    [-np.inf, 7, 28, 90, 365, np.inf],
    ["0-7 days", "8-28", "29-90", "91-365", "366+"],
    "03_enrollment_date_to_member_tenure",
    "Original Enrollment Date -> Member Tenure"
)


# ==========================================
# 4. ORIGINAL CATEGORICAL FEATURES
# ==========================================

category_graph(
    df, "booking_channel",
    "04_booking_channel",
    "Fraud Rate by Original Booking Channel"
)

category_graph(
    df, "rate_code",
    "05_rate_code",
    "Fraud Rate by Original Rate Code"
)

category_graph(
    df, "qualification_code",
    "06_qualification_code",
    "Fraud Rate by Original Qualification Code"
)

category_graph(
    df, "site_brand",
    "07_site_brand",
    "Fraud Rate by Hotel Brand"
)

category_graph(
    df, "member_enrollment_source",
    "08_member_enrollment_source",
    "Fraud Rate by Loyalty Enrollment Source"
)


# ==========================================
# 5. ORIGINAL NUMERIC FEATURES
# ==========================================

numeric_distribution(
    df,
    "points_earned",
    "09_points_earned",
    "Fraud Rate by Points Earned",
    [-np.inf, 0, 1000, 5000, 10000, 20000, np.inf],
    ["0 or less", "1-1K", "1K-5K", "5K-10K", "10K-20K", "20K+"]
)

numeric_distribution(
    df,
    "room_revenue",
    "10_room_revenue",
    "Fraud Rate by Room Revenue",
    [-np.inf, 0, 200, 500, 1000, 2000, np.inf],
    ["0 or less", "1-200", "200-500", "500-1K", "1K-2K", "2K+"]
)

numeric_distribution(
    df,
    "points_redeemed",
    "11_points_redeemed",
    "Fraud Rate by Points Redeemed",
    [-np.inf, 0, 1000, 5000, 10000, 20000, np.inf],
    ["0 or less", "1-1K", "1K-5K", "5K-10K", "10K-20K", "20K+"]
)


# ==========================================
# 6. CANCELLATION FIELD
# ==========================================

df["cancellation_status"] = np.where(
    df["cancellation_date"].notna(),
    "Cancellation recorded",
    "No cancellation recorded"
)

category_graph(
    df,
    "cancellation_status",
    "12_cancellation_status",
    "Fraud Rate by Cancellation Status (Timing Risk)"
)


# ==========================================
# 7. PROPERTY CONCENTRATION
# ==========================================

# Show properties with the most labeled fraud events,
# with total reservation counts for context.

properties = df.groupby(
    "site_id", dropna=False
)[TARGET].agg(["sum", "count"])

properties = properties.sort_values(
    ["sum", "count"], ascending=False
).head(15)

fig, ax = plt.subplots(figsize=(11, 7))

ax.barh(
    properties.index.astype(str),
    properties["sum"],
    color="#3977b8"
)

ax.invert_yaxis()
ax.set_xlabel("Fraud events")
ax.set_ylabel("Property ID")
ax.set_title("Properties with Most Fraud Events")

for i, (_, row) in enumerate(properties.iterrows()):
    ax.text(
        row["sum"] + 0.1,
        i,
        f'{int(row["sum"]):,} fraud / '
        f'{int(row["count"]):,} reservations',
        va="center",
        fontsize=9
    )

ax.set_xlim(0, max(properties["sum"].max() * 1.8, 1))

save("13_property_fraud_concentration")


# ==========================================
# 8. MEMBER BOOKING HISTORY
# ==========================================

# Count bookings earlier than the current booking
# within the May dataset only.
#
# This is an exploratory May-only count, NOT the
# full historical total_reservations feature used
# in the model. Earlier months may be needed.
#
# Reservations at the same timestamp do not
# count each other as prior history.

history = df[
    ["member_number", "booking_timestamp", TARGET]
].copy()

history = history.dropna(
    subset=["member_number", "booking_timestamp"]
)

history = history.sort_values(
    ["member_number", "booking_timestamp"]
)

same_time = history.groupby(
    ["member_number", "booking_timestamp"],
    sort=False
).size().rename("at_timestamp").reset_index()

same_time["prior_may_bookings"] = (
    same_time.groupby("member_number")["at_timestamp"]
    .cumsum()
    - same_time["at_timestamp"]
)

history = history.merge(
    same_time[
        ["member_number", "booking_timestamp",
         "prior_may_bookings"]
    ],
    on=["member_number", "booking_timestamp"],
    how="left",
    validate="many_to_one"
)

binned_graph(
    history,
    history["prior_may_bookings"],
    [-np.inf, 0, 1, 2, 5, np.inf],
    ["0", "1", "2", "3-5", "6+"],
    "14_member_booking_history",
    "Member ID + Booking Time -> Earlier May Bookings"
)


# ==========================================
# 9. FINAL SUMMARY
# ==========================================

print("\nDONE!")
print("Graphs saved to:", OUTPUT.resolve())
