import pandas as pd

data = pd.read_parquet("data/revision4_combined_cleaned.parquet")

# only members
members = data[data["member_number"].notna()].copy()

# memberlevel QNS/QXY behavior
member_counts = members.groupby("member_number").agg(
    total_reservations=("confirmation_number", "count"),
    qns_count=("qualification_code", lambda x: (x == "QNS").sum()),
    qxy_count=("qualification_code", lambda x: (x == "QXY").sum())
)

member_counts["qns_qxy_count"] = (
    member_counts["qns_count"] + member_counts["qxy_count"]
)

member_counts["qns_qxy_rate"] = (
    member_counts["qns_qxy_count"]
    / member_counts["total_reservations"]
)

# numberof months member had QNS/QXY
qns_qxy = members[
    members["qualification_code"].isin(["QNS", "QXY"])
]

member_months = (
    qns_qxy.groupby("member_number")["month"]
    .nunique()
)

member_counts["months_with_qns_qxy"] = (
    member_counts.index.map(member_months).fillna(0)
)

#loading actual fraud labels
fraud_labels = pd.read_csv(
    "data/revision4/member_ground_truth_may2026.csv"
)

fraud_labels["member_number"] = (
    fraud_labels["member_number"].astype("string")
)

member_counts = member_counts.reset_index()

member_counts["member_number"] = (
    member_counts["member_number"].astype("string")
)

member_counts = member_counts.merge(
    fraud_labels,
    on="member_number",
    how="left"
)

print("Members:", len(member_counts))

print("\nFraud labels:")
print(member_counts["is_fraud_account"].value_counts())

print("\nMembers with QNS/QXY:")
print((member_counts["qns_qxy_count"] > 0).sum())

summary = member_counts.groupby("is_fraud_account").agg(
    total_reservations=("total_reservations", "mean"),
    qns_count=("qns_count", "mean"),
    qxy_count=("qxy_count", "mean"),
    qns_qxy_count=("qns_qxy_count", "mean"),
    qns_qxy_rate=("qns_qxy_rate", "mean"),
    months_with_qns_qxy=("months_with_qns_qxy", "mean")
)

print("\nQNS/QXY behavior by fraud label:")
print(summary.to_string())

member_counts.to_parquet(
    "data/revision4_qns_qxy_member_analysis.parquet",
    index=False
)

print("\nMember analysis saved.")