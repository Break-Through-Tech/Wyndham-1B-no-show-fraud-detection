import pandas as pd

data = pd.read_parquet(
    "data/may_modeling_data.parquet",
    columns=[
        "is_fraud_event",
        "booking_lead_days",
        "member_tenure_days"
    ]
)

print("\nBOOKING LEAD DAYS BY FRAUD LABEL")

print(
    data.groupby("is_fraud_event")["booking_lead_days"]
    .describe()
)

print("\nRESERVATIONS BOOKED WITHIN 3 DAYS")

short_lead = data["booking_lead_days"] <= 3

print(
    pd.crosstab(
        data["is_fraud_event"],
        short_lead,
        rownames=["Fraud"],
        colnames=["Lead time <= 3 days"]
    )
)

print("\nMEMBER TENURE BY FRAUD LABEL")

print(
    data.groupby("is_fraud_event")["member_tenure_days"]
    .describe()
)

print("\nSHORT LEAD + NEW MEMBER COMBINATION")

rule = (
    (data["booking_lead_days"] <= 3) &
    (data["member_tenure_days"] <= 28)
)

print(
    pd.crosstab(
        data["is_fraud_event"],
        rule,
        rownames=["Fraud"],
        colnames=["Matches both conditions"]
    )
)