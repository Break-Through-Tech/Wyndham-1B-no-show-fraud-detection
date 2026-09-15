import pandas as pd

data = pd.read_parquet("data/combined_cleaned.parquet")

members = data[data["member_number"].notna()].copy()

member_counts = members.groupby("member_number").agg(
    total_reservations=("confirmation_number", "count"),
    qns_count=("qualification_code", lambda x: (x == "QNS").sum()),
    qxy_count=("qualification_code", lambda x: (x == "QXY").sum())
)

member_counts["qns_qxy_count"] = (
    member_counts["qns_count"] + member_counts["qxy_count"]
)

print(member_counts.head())

print("\nMembers:", len(member_counts))
print("Members with QNS/QXY:", (member_counts["qns_qxy_count"] > 0).sum())

print("\nQNS/QXY events per affected member:")
print(
    member_counts[member_counts["qns_qxy_count"] > 0]
    ["qns_qxy_count"]
    .value_counts()
    .sort_index()
)

#THe QNS/QXY events fro certain memebers jump from 6 to 12 which is SUS
#checking if this happens across all months

qns_qxy = members[
    members["qualification_code"].isin(["QNS", "QXY"])
]

member_months = (
    qns_qxy.groupby("member_number")["month"]
    .nunique()
)

print("\nNumber of months with QNS/QXY activity:")
print(member_months.value_counts().sort_index())

high_repeat = member_counts[
    member_counts["qns_qxy_count"] >= 6
].copy()

high_repeat["months_with_qns_qxy"] = member_months

print("\nMembers with 6+ QNS/QXY events:")
print(
    high_repeat[
        ["total_reservations", "qns_count", "qxy_count",
         "qns_qxy_count", "months_with_qns_qxy"]
    ]
    .head(20)
)

member_counts["qns_qxy_rate"] = (
    member_counts["qns_qxy_count"]
    / member_counts["total_reservations"]
)

affected = member_counts[
    member_counts["qns_qxy_count"] > 0
]

print("\nQNS/QXY rate for affected members:")
print(affected["qns_qxy_rate"].describe())

print("\nMembers with 6+ QNS/QXY:")
print(
    member_counts[member_counts["qns_qxy_count"] >= 6]
    [
        ["total_reservations", "qns_count",
         "qxy_count", "qns_qxy_count", "qns_qxy_rate"]
    ]
    .sort_values("qns_qxy_rate", ascending=False)
    .head(20)
)

qxy_members = member_counts[member_counts["qxy_count"] > 0]

print("\nMembers with QXY:", len(qxy_members))

print("\nQXY members - QNS/QXY count:")
print(qxy_members["qns_qxy_count"].value_counts().sort_index())

print("\nQXY members - QNS/QXY rate:")
print(qxy_members["qns_qxy_rate"].describe())

#checking if the 6-12 event groups are different behviroial groups
for count in [6, 12]:
    group = member_counts[member_counts["qns_qxy_count"] == count]

    print(f"\nMembers with {count} QNS/QXY events:")
    print("Members:", len(group))
    print("Total reservations:")
    print(group["total_reservations"].value_counts().sort_index())
    print("QNS counts:")
    print(group["qns_count"].value_counts().sort_index())
    print("QXY counts:")
    print(group["qxy_count"].value_counts().sort_index())