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

#notes:
'''
May, June, July have the same structure with 3.2M reservation rows and 3.2M stay rows
'''

for month in months:
    members = pd.read_csv(base + f"member_ground_truth_{month}2026.csv")
    events = pd.read_csv(base + f"fraud_event_ground_truth_{month}2026.csv")

    print(f"\n===== {month.upper()} FRAUD CHECK =====")

    print("Fraud accounts:")
    print(members["is_fraud_account"].value_counts())

    print("\nFraud mechanisms:")
    print(events["fraud_mechanism"].value_counts())

#notes:
'''
Each month's member_ground_truth had 700400 loyalty members with
700,000 legit (is_fraud_account = 0)
400 fraud (is_fraud_account = 1)

is_fraud_account is our target label for the member level model.

Each month's fraud_event_ground_truth has 1,600 known fraud events:
1,280 NO_SHOW_POINTS_HARVEST
320 LATE_CANCEL_REVENUE_RETAINED

member_ground_truth tells us WHICH MEMBERS are fraud.
fraud_event_ground_truth tells us WHICH SPECIFIC EVENTS are fraud and what fraud mechanism was used.
'''