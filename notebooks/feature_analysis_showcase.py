
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================
# 1. LOAD DATA
# ==========================================

DATA_PATH = Path("data/may_modeling_data.parquet")
OUTPUT_DIR = Path("outputs/feature_analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_PATH)

TARGET = "is_fraud_event"

print("Total reservations:", len(df))
print("Fraud reservations:", int(df[TARGET].sum()))
print("Legitimate reservations:", int((df[TARGET] == 0).sum()))


# ==========================================
# 2. FEATURES TO ANALYZE
# ==========================================

# All 18 features used across Model 1 and Model 2,
# plus historical redemption features that were excluded.

numeric_features = [
    "booking_lead_days",
    "stay_length",
    "same_day_booking",
    "member_tenure_days",
    "total_reservations",
    "single_night_ratio",
    "same_day_booking_count",
    "qns_count",
    "qxy_count",
    "qns_qxy_rate",
    "qns_qxy_points_earned",
    "points_earned",
    "room_revenue",
    "current_qns",
    "current_qxy",
    "redemption_count",
    "total_points_redeemed"
]

categorical_features = [
    "rate_code",
    "booking_channel",
    "qualification_code"
]

# Binary variables should be graphed as categories,
# not as continuous numeric distributions.
binary_features = [
    "same_day_booking",
    "current_qns",
    "current_qxy"
]

numeric_features = [
    feature
    for feature in numeric_features
    if feature not in binary_features
]

categorical_features += binary_features


# ==========================================
# 3. FEATURE STATUS
# ==========================================

# These are current decisions, not final conclusions.
# A feature can be useful statistically but unavailable
# at the time a fraud prediction must be made.

feature_status = {
    "booking_lead_days": "Keep provisionally",
    "stay_length": "Evaluate",
    "same_day_booking": "Evaluate",
    "rate_code": "Evaluate",
    "booking_channel": "Evaluate",
    "member_tenure_days": "Keep provisionally",
    "total_reservations": "Evaluate",
    "single_night_ratio": "Evaluate",
    "same_day_booking_count": "Evaluate",
    "qns_count": "Timing unconfirmed",
    "qxy_count": "Timing unconfirmed",
    "qns_qxy_rate": "Timing unconfirmed",
    "qns_qxy_points_earned": "Timing unconfirmed",
    "qualification_code": "Timing unconfirmed",
    "points_earned": "Timing unconfirmed",
    "room_revenue": "Timing unconfirmed",
    "current_qns": "Timing unconfirmed",
    "current_qxy": "Timing unconfirmed",
    "redemption_count": "Excluded: timestamps unavailable",
    "total_points_redeemed": "Excluded: timestamps unavailable"
}


# ==========================================
# 4. GRAPH FUNCTIONS
# ==========================================

def save_numeric_graph(feature):

    # Drop missing values only for this graph.
    subset = df[[feature, TARGET]].dropna().copy()

    fraud = subset.loc[
        subset[TARGET] == 1, feature
    ]

    legitimate = subset.loc[
        subset[TARGET] == 0, feature
    ]

    if fraud.empty or legitimate.empty:
        print("Skipped:", feature, "- insufficient data")
        return

    # Clip the display at the 99th percentile so that
    # extreme outliers do not make graphs unreadable.
    # The underlying data is not changed.
    upper = subset[feature].quantile(0.99)
    lower = subset[feature].quantile(0.01)

    if upper <= lower:
        lower = subset[feature].min()
        upper = subset[feature].max()

    if upper <= lower:
        print("Skipped:", feature, "- no variation")
        return

    plt.figure(figsize=(10, 5))

    plt.hist(
        legitimate.clip(lower, upper),
        bins=35,
        density=True,
        alpha=0.5,
        label="Legitimate"
    )

    plt.hist(
        fraud.clip(lower, upper),
        bins=35,
        density=True,
        alpha=0.6,
        label="Fraud"
    )

    plt.title(
        f"{feature.replace('_', ' ').title()} - Fraud vs Legitimate"
    )

    plt.xlabel(feature.replace("_", " ").title())
    plt.ylabel("Normalized density")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{feature}.png",
        dpi=160
    )

    plt.close()


def save_categorical_graph(feature):

    subset = df[[feature, TARGET]].copy()

    subset[feature] = (
        subset[feature]
        .astype("string")
        .fillna("Missing")
    )

    grouped = subset.groupby(feature)[TARGET].agg(
        fraud_count="sum",
        total_count="count"
    )

    grouped["fraud_rate"] = (
        grouped["fraud_count"] /
        grouped["total_count"] * 100
    )

    # Display the 15 most common categories.
    grouped = grouped.sort_values(
        "total_count", ascending=False
    ).head(15)

    grouped = grouped.sort_values("fraud_rate")

    plt.figure(figsize=(10, 5))

    plt.barh(
        grouped.index.astype(str),
        grouped["fraud_rate"]
    )

    plt.title(
        f"Fraud Rate by {feature.replace('_', ' ').title()}"
    )

    plt.xlabel("Fraud rate (%)")
    plt.ylabel(feature.replace("_", " ").title())

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{feature}.png",
        dpi=160
    )

    plt.close()


# ==========================================
# 5. GENERATE ALL GRAPHS
# ==========================================

print("\nGENERATING NUMERIC FEATURE GRAPHS")

for feature in numeric_features:

    if feature not in df.columns:
        print("Not found:", feature)
        continue

    save_numeric_graph(feature)
    print("Processed:", feature)


print("\nGENERATING CATEGORICAL FEATURE GRAPHS")

for feature in categorical_features:

    if feature not in df.columns:
        print("Not found:", feature)
        continue

    save_categorical_graph(feature)
    print("Processed:", feature)


# ==========================================
# 6. GENERATE FEATURE SUMMARY TABLE
# ==========================================

summary_rows = []

for feature in numeric_features + categorical_features:

    if feature not in df.columns:
        summary_rows.append({
            "Feature": feature,
            "Type": "Unavailable",
            "Legitimate Mean": np.nan,
            "Fraud Mean": np.nan,
            "Missing %": np.nan,
            "Status": feature_status.get(feature, "Evaluate")
        })
        continue

    is_numeric = feature in numeric_features

    if is_numeric:
        legitimate_mean = df.loc[
            df[TARGET] == 0, feature
        ].mean()

        fraud_mean = df.loc[
            df[TARGET] == 1, feature
        ].mean()
    else:
        legitimate_mean = np.nan
        fraud_mean = np.nan

    summary_rows.append({
        "Feature": feature,
        "Type": "Numeric" if is_numeric else "Categorical",
        "Legitimate Mean": legitimate_mean,
        "Fraud Mean": fraud_mean,
        "Missing %": df[feature].isna().mean() * 100,
        "Status": feature_status.get(feature, "Evaluate")
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUTPUT_DIR / "feature_summary.csv",
    index=False
)

print("\nFEATURE SUMMARY")
print(summary.to_string(index=False))

print("\nDONE!")
print("Graphs and summary saved to:", OUTPUT_DIR.resolve())

# ==========================================
# 7. ADDITIONAL FEATURE-SELECTION GRAPHS
# ==========================================

print("\nGENERATING FEATURE RELATIONSHIP GRAPHS")


# GRAPH 1: BOOKING LEAD TIME × MEMBER TENURE
# Shows fraud rates for combinations of booking timing
# and loyalty account age.

lead_bins = [-float("inf"), 3, 7, 30, float("inf")]
lead_labels = ["0-3 days", "4-7 days", "8-30 days", "31+ days"]

tenure_bins = [-float("inf"), 28, 90, 365, float("inf")]
tenure_labels = [
    "0-28 days",
    "29-90 days",
    "91-365 days",
    "366+ days"
]

relationship_df = df[
    ["booking_lead_days", "member_tenure_days", TARGET]
].dropna().copy()

relationship_df["Lead Time"] = pd.cut(
    relationship_df["booking_lead_days"],
    bins=lead_bins,
    labels=lead_labels
)

relationship_df["Member Tenure"] = pd.cut(
    relationship_df["member_tenure_days"],
    bins=tenure_bins,
    labels=tenure_labels
)

lead_tenure = relationship_df.pivot_table(
    index="Lead Time",
    columns="Member Tenure",
    values=TARGET,
    aggfunc="mean",
    observed=False
) * 100

lead_tenure_counts = relationship_df.pivot_table(
    index="Lead Time",
    columns="Member Tenure",
    values=TARGET,
    aggfunc="count",
    observed=False
).fillna(0)

fig, ax = plt.subplots(figsize=(10, 6))

heat = ax.imshow(
    lead_tenure.to_numpy(dtype=float),
    cmap="Blues",
    aspect="auto"
)

ax.set_xticks(range(len(lead_tenure.columns)))
ax.set_xticklabels(lead_tenure.columns)

ax.set_yticks(range(len(lead_tenure.index)))
ax.set_yticklabels(lead_tenure.index)

ax.set_xlabel("Member Tenure")
ax.set_ylabel("Booking Lead Time")
ax.set_title("Fraud Rate by Booking Lead Time and Member Tenure")

for i in range(len(lead_tenure.index)):
    for j in range(len(lead_tenure.columns)):
        value = lead_tenure.iloc[i, j]
        count = int(lead_tenure_counts.iloc[i, j])

        label = (
            f"{value:.2f}%\nn={count:,}"
            if count > 0 and pd.notna(value)
            else "No data"
        )

        ax.text(
            j, i, label,
            ha="center", va="center",
            fontsize=8
        )

fig.colorbar(heat, ax=ax, label="Fraud Rate (%)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "relationship_lead_time_member_tenure.png",
    dpi=160
)
plt.close()


# GRAPH 2: BOOKING CHANNEL × RATE CODE
# Tests whether certain combinations are more
# associated with fraud than individual categories.

channel_rate = df.groupby(
    ["booking_channel", "rate_code"],
    dropna=False
)[TARGET].agg(
    fraud_count="sum",
    total_count="count"
).reset_index()

channel_rate["fraud_rate"] = (
    channel_rate["fraud_count"] /
    channel_rate["total_count"] * 100
)

rate_matrix = channel_rate.pivot(
    index="booking_channel",
    columns="rate_code",
    values="fraud_rate"
)

count_matrix = channel_rate.pivot(
    index="booking_channel",
    columns="rate_code",
    values="total_count"
).reindex(
    index=rate_matrix.index,
    columns=rate_matrix.columns
)

fig, ax = plt.subplots(figsize=(15, 7))

heat = ax.imshow(
    rate_matrix.to_numpy(dtype=float),
    cmap="Blues",
    aspect="auto"
)

ax.set_xticks(range(len(rate_matrix.columns)))
ax.set_xticklabels(rate_matrix.columns, rotation=45, ha="right")

ax.set_yticks(range(len(rate_matrix.index)))
ax.set_yticklabels(rate_matrix.index)

ax.set_xlabel("Rate Code")
ax.set_ylabel("Booking Channel")
ax.set_title("Fraud Rate by Booking Channel and Rate Code")

for i in range(len(rate_matrix.index)):
    for j in range(len(rate_matrix.columns)):
        value = rate_matrix.iloc[i, j]
        count = count_matrix.iloc[i, j]

        if pd.notna(value) and pd.notna(count):
            ax.text(
                j, i,
                f"{value:.2f}%\nn={int(count):,}",
                ha="center",
                va="center",
                fontsize=6
            )

fig.colorbar(heat, ax=ax, label="Fraud Rate (%)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "relationship_channel_rate_code.png",
    dpi=160
)
plt.close()


# GRAPH 3: NUMERIC FEATURE CORRELATION
# Measures relationships between numeric features.
# Strong correlation can suggest overlapping information,
# but does not automatically mean a feature should be removed.

correlation_features = [
    "booking_lead_days",
    "stay_length",
    "same_day_booking",
    "member_tenure_days",
    "total_reservations",
    "single_night_ratio",
    "same_day_booking_count",
    "qns_count",
    "qxy_count",
    "qns_qxy_rate",
    "qns_qxy_points_earned",
    "points_earned",
    "room_revenue",
    "current_qns",
    "current_qxy"
]

available_correlation_features = [
    feature
    for feature in correlation_features
    if feature in df.columns
]

correlation_data = df[
    available_correlation_features
].apply(pd.to_numeric, errors="coerce")

correlation_matrix = correlation_data.corr()

fig, ax = plt.subplots(figsize=(14, 11))

heat = ax.imshow(
    correlation_matrix.to_numpy(),
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

ax.set_xticks(range(len(correlation_matrix.columns)))
ax.set_xticklabels(
    correlation_matrix.columns,
    rotation=90,
    fontsize=8
)

ax.set_yticks(range(len(correlation_matrix.index)))
ax.set_yticklabels(
    correlation_matrix.index,
    fontsize=8
)

ax.set_title("Correlation Between Engineered and Original Features")

fig.colorbar(heat, ax=ax, label="Pearson Correlation")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_correlation_matrix.png",
    dpi=160
)
plt.close()

correlation_matrix.to_csv(
    OUTPUT_DIR / "feature_correlation_matrix.csv"
)


# GRAPH 4: MODEL FEATURE-GROUP COMPARISON
# Uses your actual member-based test results.

model_comparison = pd.DataFrame({
    "Model": [
        "Model 1\n13 features",
        "Conservative\n9 features",
        "Model 2\n18 features",
        "Simple rule\n2 features"
    ],
    "False Positives": [326, 357, 8, 235],
    "Precision": [0.489827856, 0.467164179, 0.975077882, 0.571167883],
    "Recall": [1.0, 1.0, 1.0, 1.0]
})

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    model_comparison["Model"],
    model_comparison["False Positives"]
)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 5,
        str(int(height)),
        ha="center"
    )

ax.set_ylabel("False Positive Reservations")
ax.set_title("False Positives by Feature Set - May Test Members")
ax.set_ylim(0, 400)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_feature_group_comparison.png",
    dpi=160
)
plt.close()

model_comparison.to_csv(
    OUTPUT_DIR / "model_feature_group_comparison.csv",
    index=False
)

print("\nADDITIONAL GRAPHS COMPLETE")
print("Saved to:", OUTPUT_DIR.resolve())

# GRAPH 5: MODEL 1 VS MODEL 2

model_1_vs_2 = model_comparison.iloc[[0, 2]].copy()

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    ["Model 1\n13 features", "Model 2\n18 features"],
    model_1_vs_2["False Positives"],
    color=["#94A3B8", "#2563EB"],
    width=0.5
)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 5,
        str(int(height)),
        ha="center",
        fontsize=12
    )

ax.set_title("False Fraud Alerts: Model 1 vs Model 2")
ax.set_ylabel("False Positive Reservations")
ax.set_ylim(0, 380)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_1_vs_model_2.png",
    dpi=160
)
plt.close()