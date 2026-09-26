import pandas as pd
pd.set_option("display.max_columns", None) #tryna see the whole set of columns
#loading points/redemption member analysis
data = pd.read_parquet(
    "data/member_points_redemption_analysis.parquet"
)

fraud_labels = pd.read_csv(
    "data/revision4/member_ground_truth_may2026.csv"
)

data = data.merge(
    fraud_labels,
    on="member_number",
    how="left"
)

features = [
    "qns_qxy_points_earned",
    "max_points_single_qns_qxy",
    "redemption_count",
    "total_points_redeemed",
    "days_to_redemption"
]

print("\nFRAUD VS LEGIT - POINTS + REDEMPTION")
print(data.groupby("is_fraud_account")[features].mean())