import pandas as pd

base = "data/revision4/"

months = ["may", "june", "july"]

for month in months:
    files = [
        f"reservations_{month}_2026.csv",
        f"stays_{month}_2026.csv",
        f"member_ground_truth_{month}2026.csv",
        f"fraud_event_ground_truth_{month}2026.csv",
    ]

    print(f"\n===== {month.upper()} =====")

    for file in files:
        df = pd.read_csv(base + file)

        print("\n", file)
        print("Shape:", df.shape)
        print("Columns:", list(df.columns))

for month in months:
    members = pd.read_csv(base + f"member_ground_truth_{month}2026.csv")
    events = pd.read_csv(base + f"fraud_event_ground_truth_{month}2026.csv")

    print(f"\n===== {month.upper()} FRAUD CHECK =====")

    print("Fraud accounts:")
    print(members["is_fraud_account"].value_counts())

    print("\nFraud mechanisms:")
    print(events["fraud_mechanism"].value_counts())