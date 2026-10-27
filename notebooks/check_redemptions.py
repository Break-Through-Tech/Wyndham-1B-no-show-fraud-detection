import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data = pd.read_parquet(
    "data/revision4_combined_cleaned.parquet",
    columns=["month", "confirmation_number", "points_redeemed"]
)

fraud = pd.read_csv(
    "data/revision4/fraud_event_ground_truth_may2026.csv"
)

may = data[data["month"] == "May"].copy()

may["is_fraud_event"] = may["confirmation_number"].isin(
    fraud["confirmation_number"]
).astype(int)

# Convert negative redemption values into positive amounts
may["redeemed_points"] = may["points_redeemed"].abs()

# Group reservations by redemption amount
may["redemption_group"] = pd.cut(
    may["redeemed_points"],
    bins=[-1, 0, 5000, 15000, 30000, float("inf")],
    labels=["0", "1–5K", "5K–15K", "15K–30K", "30K+"]
)

summary = (
    may.groupby("redemption_group", observed=False)
    .agg(
        reservations=("confirmation_number", "size"),
        fraud_events=("is_fraud_event", "sum")
    )
)

summary["fraud_rate_pct"] = (
    summary["fraud_events"] / summary["reservations"].replace(0, float("nan")) * 100
).round(4)

print(summary.to_string())


# Create Chart #11 using the corrected redemption data
labels = summary.index.astype(str).tolist()
rates = summary["fraud_rate_pct"].tolist()
counts = summary["reservations"].tolist()
fraud_counts = summary["fraud_events"].tolist()

fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.bar(labels, rates, color="#4F79B5", width=0.65)

ax.set_title("Fraud Rate by Points Redeemed", fontsize=15, fontweight="bold")
ax.set_xlabel("Points Redeemed")
ax.set_ylabel("Fraud rate (%)")
ax.set_ylim(0, max(rates) * 1.25)

# Add fraud rates and reservation counts above bars
for bar, rate, fraud_count, count in zip(
    bars, rates, fraud_counts, counts
):
    ax.annotate(
        f"{rate:.4f}%\n({fraud_count:,}/{count:,})",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9
    )

ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

plt.tight_layout()

# Save the chart
output = Path("charts")
output.mkdir(exist_ok=True)

plt.savefig(output / "chart_11_points_redeemed_corrected.png", dpi=300)
plt.show()

print("\nSaved chart to charts/chart_11_points_redeemed_corrected.png")