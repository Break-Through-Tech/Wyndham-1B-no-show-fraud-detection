import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_parquet("data/combined_may_june_july.parquet")

print("Dataset shape:", data.shape)

print("\nColumns:")
print(data.columns.tolist())


# ============================================================
# 2. IMPORTANT COLUMNS
# ============================================================

MEMBER_COL = "member_number"
QUALIFICATION_COL = "qualification_code"
RATE_COL = "rate_code"

POINTS_EARNED_COL = "points_earned"
POINTS_REDEEMED_COL = "points_redeemed"

DATE_COL = "booking_timestamp"


# ============================================================
# 3. BASIC CLEANING
# ============================================================

data[DATE_COL] = pd.to_datetime(
    data[DATE_COL],
    errors="coerce"
)

data[POINTS_EARNED_COL] = pd.to_numeric(
    data[POINTS_EARNED_COL],
    errors="coerce"
)

data[POINTS_REDEEMED_COL] = pd.to_numeric(
    data[POINTS_REDEEMED_COL],
    errors="coerce"
)


# ============================================================
# 4. CHECK RELEVANT CODES
# ============================================================

print("\n====================================")
print("CODE CHECK")
print("====================================")

print("\nQualification codes:")
print(
    data[QUALIFICATION_COL]
    .value_counts(dropna=False)
    .head(20)
)

print("\nRate codes:")
print(
    data[RATE_COL]
    .value_counts(dropna=False)
    .head(20)
)


# ============================================================
# 5. CREATE QNS / QXY / SRB DATASETS
# ============================================================

# QNS and QXY are qualification codes
qns_qxy = data[
    data[QUALIFICATION_COL].isin(
        ["QNS", "QXY"]
    )
].copy()


# SRB is a rate code
srb = data[
    data[RATE_COL] == "SRB"
].copy()


print("\n====================================")
print("BASIC COUNTS")
print("====================================")

print(
    "QNS records:",
    (data[QUALIFICATION_COL] == "QNS").sum()
)

print(
    "QXY records:",
    (data[QUALIFICATION_COL] == "QXY").sum()
)

print(
    "Total QNS/QXY records:",
    len(qns_qxy)
)

print(
    "SRB records:",
    len(srb)
)

print(
    "Unique QNS/QXY members:",
    qns_qxy[MEMBER_COL].nunique()
)

print(
    "Unique SRB members:",
    srb[MEMBER_COL].nunique()
)


# ============================================================
# 6. MEMBER-LEVEL QNS/QXY SUMMARY
# ============================================================

qns_qxy_summary = (
    qns_qxy
    .groupby(MEMBER_COL)
    .agg(
        qns_qxy_count=(
            QUALIFICATION_COL,
            "size"
        ),

        qns_qxy_points_earned=(
            POINTS_EARNED_COL,
            "sum"
        ),

        avg_points_per_qns_qxy=(
            POINTS_EARNED_COL,
            "mean"
        ),

        max_points_single_qns_qxy=(
            POINTS_EARNED_COL,
            "max"
        )
    )
    .reset_index()
)


print("\n====================================")
print("QNS/QXY MEMBER SUMMARY")
print("====================================")

print(qns_qxy_summary.head())


# ============================================================
# 7. QNS AND QXY COUNTS SEPARATELY
# ============================================================

qns_counts = (
    qns_qxy[
        qns_qxy[QUALIFICATION_COL] == "QNS"
    ]
    .groupby(MEMBER_COL)
    .size()
    .reset_index(
        name="qns_count"
    )
)


qxy_counts = (
    qns_qxy[
        qns_qxy[QUALIFICATION_COL] == "QXY"
    ]
    .groupby(MEMBER_COL)
    .size()
    .reset_index(
        name="qxy_count"
    )
)


qns_qxy_summary = qns_qxy_summary.merge(
    qns_counts,
    on=MEMBER_COL,
    how="left"
)


qns_qxy_summary = qns_qxy_summary.merge(
    qxy_counts,
    on=MEMBER_COL,
    how="left"
)


qns_qxy_summary[
    ["qns_count", "qxy_count"]
] = (
    qns_qxy_summary[
        ["qns_count", "qxy_count"]
    ]
    .fillna(0)
    .astype(int)
)


# ============================================================
# 8. QNS/QXY POINTS DISTRIBUTION
# ============================================================

print("\n====================================")
print("QNS/QXY POINTS DISTRIBUTION")
print("====================================")

print(
    qns_qxy_summary[
        "qns_qxy_points_earned"
    ].describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            0.999
        ]
    )
)


# ============================================================
# 9. FLAG HIGH QNS/QXY POINT EARNERS
# ============================================================

points_99_threshold = (
    qns_qxy_summary[
        "qns_qxy_points_earned"
    ].quantile(0.99)
)


qns_qxy_summary[
    "high_qns_qxy_points"
] = (
    qns_qxy_summary[
        "qns_qxy_points_earned"
    ]
    >= points_99_threshold
).astype(int)


print(
    "\n99th percentile points threshold:",
    points_99_threshold
)

print(
    "Members above threshold:",
    qns_qxy_summary[
        "high_qns_qxy_points"
    ].sum()
)


# ============================================================
# 10. FLAG HIGH QNS/QXY EVENT COUNTS
# ============================================================

event_99_threshold = (
    qns_qxy_summary[
        "qns_qxy_count"
    ].quantile(0.99)
)


qns_qxy_summary[
    "high_qns_qxy_count"
] = (
    qns_qxy_summary[
        "qns_qxy_count"
    ]
    >= event_99_threshold
).astype(int)


print(
    "\n99th percentile event-count threshold:",
    event_99_threshold
)

print(
    "Members above event threshold:",
    qns_qxy_summary[
        "high_qns_qxy_count"
    ].sum()
)


# ============================================================
# 11. REDEMPTION SUMMARY
# ============================================================

redemption_summary = (
    srb
    .groupby(MEMBER_COL)
    .agg(
        redemption_count=(
            RATE_COL,
            "size"
        ),

        total_points_redeemed=(
            POINTS_REDEEMED_COL,
            "sum"
        ),

        avg_points_redeemed=(
            POINTS_REDEEMED_COL,
            "mean"
        ),

        max_points_redeemed=(
            POINTS_REDEEMED_COL,
            "max"
        )
    )
    .reset_index()
)


print("\n====================================")
print("REDEMPTION SUMMARY")
print("====================================")

print(
    redemption_summary[
        [
            "redemption_count",
            "total_points_redeemed",
            "avg_points_redeemed",
            "max_points_redeemed"
        ]
    ].describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)


# ============================================================
# 12. COMBINE QNS/QXY + REDEMPTION
# ============================================================

member_behavior = qns_qxy_summary.merge(
    redemption_summary,
    on=MEMBER_COL,
    how="left"
)


redemption_cols = [
    "redemption_count",
    "total_points_redeemed",
    "avg_points_redeemed",
    "max_points_redeemed"
]


member_behavior[
    redemption_cols
] = (
    member_behavior[
        redemption_cols
    ]
    .fillna(0)
)


# ============================================================
# 13. DID QNS/QXY MEMBERS ALSO REDEEM?
# ============================================================

member_behavior[
    "qns_qxy_member_redeemed"
] = (
    member_behavior[
        "redemption_count"
    ] > 0
).astype(int)


members_with_qns_qxy = len(
    member_behavior
)


members_who_redeemed = (
    member_behavior[
        "qns_qxy_member_redeemed"
    ].sum()
)


percent_redeemed = (
    members_who_redeemed
    / members_with_qns_qxy
    * 100
)


print("\n====================================")
print("QNS/QXY MEMBERS WHO REDEEM")
print("====================================")

print(
    "Members with QNS/QXY:",
    members_with_qns_qxy
)

print(
    "QNS/QXY members who also redeemed:",
    members_who_redeemed
)

print(
    f"Percent who redeemed: "
    f"{percent_redeemed:.2f}%"
)


# ============================================================
# 14. FIRST QNS/QXY DATE
# ============================================================

first_qns_qxy = (
    qns_qxy
    .groupby(MEMBER_COL)[DATE_COL]
    .min()
    .reset_index(
        name="first_qns_qxy_date"
    )
)


# ============================================================
# 15. FIRST REDEMPTION DATE
# ============================================================

first_redemption = (
    srb
    .groupby(MEMBER_COL)[DATE_COL]
    .min()
    .reset_index(
        name="first_redemption_date"
    )
)


member_behavior = member_behavior.merge(
    first_qns_qxy,
    on=MEMBER_COL,
    how="left"
)


member_behavior = member_behavior.merge(
    first_redemption,
    on=MEMBER_COL,
    how="left"
)


# ============================================================
# 16. DID REDEMPTION HAPPEN AFTER QNS/QXY?
# ============================================================

member_behavior[
    "redeemed_after_qns_qxy"
] = (
    member_behavior[
        "first_redemption_date"
    ]
    >
    member_behavior[
        "first_qns_qxy_date"
    ]
).astype(int)


print(
    "\nMembers who redeemed after "
    "their first QNS/QXY event:"
)

print(
    member_behavior[
        "redeemed_after_qns_qxy"
    ].value_counts()
)


# ============================================================
# 17. DAYS BETWEEN QNS/QXY AND REDEMPTION
# ============================================================

member_behavior[
    "days_to_redemption"
] = (
    member_behavior[
        "first_redemption_date"
    ]
    -
    member_behavior[
        "first_qns_qxy_date"
    ]
).dt.days


valid_redemption_days = (
    member_behavior.loc[
        member_behavior[
            "days_to_redemption"
        ] >= 0,
        "days_to_redemption"
    ]
)


print("\n====================================")
print("TIME TO REDEMPTION")
print("====================================")

print(
    valid_redemption_days.describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)


# ============================================================
# 18. QUICK REDEMPTION FLAGS
# ============================================================

member_behavior[
    "redeemed_within_7_days"
] = (
    member_behavior[
        "days_to_redemption"
    ].between(0, 7)
).astype(int)


member_behavior[
    "redeemed_within_30_days"
] = (
    member_behavior[
        "days_to_redemption"
    ].between(0, 30)
).astype(int)


print(
    "\nRedeemed within 7 days:",
    member_behavior[
        "redeemed_within_7_days"
    ].sum()
)

print(
    "Redeemed within 30 days:",
    member_behavior[
        "redeemed_within_30_days"
    ].sum()
)


# ============================================================
# 19. REDEMPTION / EARNING RATIO
# ============================================================

member_behavior[
    "redemption_to_qns_qxy_points_ratio"
] = np.where(
    member_behavior[
        "qns_qxy_points_earned"
    ] > 0,

    member_behavior[
        "total_points_redeemed"
    ]
    /
    member_behavior[
        "qns_qxy_points_earned"
    ],

    np.nan
)


print("\n====================================")
print("REDEMPTION / EARNING RATIO")
print("====================================")

print(
    member_behavior[
        "redemption_to_qns_qxy_points_ratio"
    ].describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)


# ============================================================
# 20. CREATE EXPLORATORY SUSPICIOUS-BEHAVIOR SCORE
# ============================================================
# This is NOT a confirmed fraud label.
# It is only used to rank unusual behavior.

member_behavior[
    "suspicious_behavior_score"
] = (
    member_behavior[
        "high_qns_qxy_points"
    ]
    +
    member_behavior[
        "high_qns_qxy_count"
    ]
    +
    member_behavior[
        "qns_qxy_member_redeemed"
    ]
    +
    member_behavior[
        "redeemed_within_30_days"
    ]
)


print("\n====================================")
print("SUSPICIOUS BEHAVIOR SCORE")
print("====================================")

print(
    member_behavior[
        "suspicious_behavior_score"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 21. TOP MEMBERS FOR REVIEW
# ============================================================

top_members = (
    member_behavior
    .sort_values(
        [
            "suspicious_behavior_score",
            "qns_qxy_points_earned",
            "qns_qxy_count",
            "total_points_redeemed"
        ],
        ascending=False
    )
)


print("\n====================================")
print("TOP 25 MEMBERS FOR REVIEW")
print("====================================")


print(
    top_members[
        [
            MEMBER_COL,
            "qns_count",
            "qxy_count",
            "qns_qxy_count",
            "qns_qxy_points_earned",
            "avg_points_per_qns_qxy",
            "redemption_count",
            "total_points_redeemed",
            "avg_points_redeemed",
            "days_to_redemption",
            "redeemed_within_7_days",
            "redeemed_within_30_days",
            "suspicious_behavior_score"
        ]
    ].head(25)
)


# ============================================================
# VISUALIZATION 1
# QNS VS QXY COUNTS
# ============================================================

qns_qxy_counts = (
    qns_qxy[
        QUALIFICATION_COL
    ]
    .value_counts()
)


plt.figure(
    figsize=(7, 5)
)

qns_qxy_counts.plot(
    kind="bar"
)

plt.title(
    "Number of QNS and QXY Events"
)

plt.xlabel(
    "Qualification Code"
)

plt.ylabel(
    "Number of Records"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 2
# DISTRIBUTION OF QNS/QXY EVENTS PER MEMBER
# ============================================================

event_limit = (
    member_behavior[
        "qns_qxy_count"
    ].quantile(0.99)
)


event_plot = member_behavior[
    member_behavior[
        "qns_qxy_count"
    ]
    <= event_limit
]


plt.figure(
    figsize=(9, 5)
)

plt.hist(
    event_plot[
        "qns_qxy_count"
    ],
    bins=30
)

plt.title(
    "Distribution of QNS/QXY Events per Member"
)

plt.xlabel(
    "Number of QNS/QXY Events"
)

plt.ylabel(
    "Number of Members"
)

plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 3
# DISTRIBUTION OF QNS/QXY POINTS EARNED
# ============================================================

points_limit = (
    member_behavior[
        "qns_qxy_points_earned"
    ].quantile(0.99)
)


points_plot = member_behavior[
    member_behavior[
        "qns_qxy_points_earned"
    ]
    <= points_limit
]


plt.figure(
    figsize=(9, 5)
)

plt.hist(
    points_plot[
        "qns_qxy_points_earned"
    ],
    bins=40
)

plt.title(
    "Distribution of QNS/QXY Points Earned per Member"
)

plt.xlabel(
    "QNS/QXY Points Earned"
)

plt.ylabel(
    "Number of Members"
)

plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 4
# TOP 20 MEMBERS BY QNS/QXY POINTS EARNED
# ============================================================

top20_points = (
    member_behavior
    .nlargest(
        20,
        "qns_qxy_points_earned"
    )
    .sort_values(
        "qns_qxy_points_earned"
    )
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top20_points[
        MEMBER_COL
    ].astype(str),

    top20_points[
        "qns_qxy_points_earned"
    ]
)

plt.title(
    "Top 20 Members by QNS/QXY Points Earned"
)

plt.xlabel(
    "QNS/QXY Points Earned"
)

plt.ylabel(
    "Member Number"
)

plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 5
# TOP 20 MEMBERS BY QNS/QXY EVENT COUNT
# ============================================================

top20_events = (
    member_behavior
    .nlargest(
        20,
        "qns_qxy_count"
    )
    .sort_values(
        "qns_qxy_count"
    )
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top20_events[
        MEMBER_COL
    ].astype(str),

    top20_events[
        "qns_qxy_count"
    ]
)

plt.title(
    "Top 20 Members by QNS/QXY Event Count"
)

plt.xlabel(
    "Number of QNS/QXY Events"
)

plt.ylabel(
    "Member Number"
)

plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 6
# QNS/QXY MEMBERS WHO REDEEM VS DO NOT REDEEM
# ============================================================

redeem_counts = (
    member_behavior[
        "qns_qxy_member_redeemed"
    ]
    .value_counts()
    .reindex(
        [0, 1],
        fill_value=0
    )
)


redeem_counts.index = [
    "Did Not Redeem",
    "Redeemed"
]


plt.figure(
    figsize=(7, 5)
)

redeem_counts.plot(
    kind="bar"
)

plt.title(
    "Redemption Behavior of QNS/QXY Members"
)

plt.xlabel(
    "Redemption Behavior"
)

plt.ylabel(
    "Number of Members"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 7
# DAYS BETWEEN QNS/QXY AND REDEMPTION
# ============================================================

days_plot = member_behavior.loc[
    member_behavior[
        "days_to_redemption"
    ].between(0, 90),

    "days_to_redemption"
]


plt.figure(
    figsize=(9, 5)
)

plt.hist(
    days_plot,
    bins=30
)

plt.title(
    "Days Between QNS/QXY Activity and Redemption"
)

plt.xlabel(
    "Days Until Redemption"
)

plt.ylabel(
    "Number of Members"
)

plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 8
# QNS/QXY EVENTS VS POINTS EARNED
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    member_behavior[
        "qns_qxy_count"
    ],

    member_behavior[
        "qns_qxy_points_earned"
    ],

    alpha=0.4
)

plt.title(
    "QNS/QXY Event Frequency vs Points Earned"
)

plt.xlabel(
    "Number of QNS/QXY Events"
)

plt.ylabel(
    "QNS/QXY Points Earned"
)

plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 9
# QNS/QXY POINTS EARNED VS POINTS REDEEMED
# ============================================================

redeemers = member_behavior[
    member_behavior[
        "total_points_redeemed"
    ] > 0
].copy()


plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    redeemers[
        "qns_qxy_points_earned"
    ],

    redeemers[
        "total_points_redeemed"
    ],

    alpha=0.4
)

plt.title(
    "QNS/QXY Points Earned vs Points Redeemed"
)

plt.xlabel(
    "QNS/QXY Points Earned"
)

plt.ylabel(
    "Total Points Redeemed"
)

plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 10
# QNS/QXY EVENT COUNT VS POINTS REDEEMED
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    member_behavior[
        "qns_qxy_count"
    ],

    member_behavior[
        "total_points_redeemed"
    ],

    alpha=0.4
)

plt.title(
    "QNS/QXY Frequency vs Points Redeemed"
)

plt.xlabel(
    "Number of QNS/QXY Events"
)

plt.ylabel(
    "Total Points Redeemed"
)

plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 11
# SUSPICIOUS BEHAVIOR SCORE DISTRIBUTION
# ============================================================

score_counts = (
    member_behavior[
        "suspicious_behavior_score"
    ]
    .value_counts()
    .sort_index()
)


plt.figure(
    figsize=(7, 5)
)

score_counts.plot(
    kind="bar"
)

plt.title(
    "Distribution of Suspicious Behavior Scores"
)

plt.xlabel(
    "Suspicious Behavior Score"
)

plt.ylabel(
    "Number of Members"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()
plt.show()


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n====================================")
print("FINAL POINTS + REDEMPTION SUMMARY")
print("====================================")

print(
    "Total QNS records:",
    (data[QUALIFICATION_COL] == "QNS").sum()
)

print(
    "Total QXY records:",
    (data[QUALIFICATION_COL] == "QXY").sum()
)

print(
    "Total QNS/QXY records:",
    len(qns_qxy)
)

print(
    "Unique QNS/QXY members:",
    qns_qxy[
        MEMBER_COL
    ].nunique()
)

print(
    "Total QNS/QXY points earned:",
    qns_qxy[
        POINTS_EARNED_COL
    ].sum()
)

print(
    "Total SRB records:",
    len(srb)
)

print(
    "Unique SRB members:",
    srb[
        MEMBER_COL
    ].nunique()
)

print(
    "Total points redeemed:",
    srb[
        POINTS_REDEEMED_COL
    ].sum()
)

print(
    f"QNS/QXY members who also redeemed: "
    f"{percent_redeemed:.2f}%"
)

print(
    "Members who redeemed within 7 days:",
    member_behavior[
        "redeemed_within_7_days"
    ].sum()
)

print(
    "Members who redeemed within 30 days:",
    member_behavior[
        "redeemed_within_30_days"
    ].sum()
)

print(
    "High QNS/QXY point earners:",
    member_behavior[
        "high_qns_qxy_points"
    ].sum()
)

print(
    "High QNS/QXY event-count members:",
    member_behavior[
        "high_qns_qxy_count"
    ].sum()
)


# ============================================================
# 23. SAVE RESULTS
# ============================================================

member_behavior.to_csv(
    "data/member_points_redemption_analysis.csv",
    index=False
)

member_behavior.to_parquet(
    "data/member_points_redemption_analysis.parquet",
    index=False
)


print("\nFiles saved successfully:")

print(
    "data/member_points_redemption_analysis.csv"
)

print(
    "data/member_points_redemption_analysis.parquet"
)

