import pandas as pd
data = pd.read_parquet("data/combined_may_june_july.parquet")
print(data.shape)

#im coverting numerical/date cols to datetime
#Note to self: booking_timestamp has timezone
date_columns = ["booking_timestamp","check_in_date","check_out_date","cancellation_date","member_enrollment_date"]
for col in date_columns:
    data[col] = pd.to_datetime(data[col])

print(data[date_columns].dtypes)

#checking for weird checkin and checkouts like (checkingout before chekcing in)
#and booking after checking in to identify potential bad actors

print("Checkout before check in:", (data["check_out_date"] < data["check_in_date"]).sum())
print("Booking date after check in date:",(data["booking_timestamp"].dt.date > data["check_in_date"].dt.date).sum())

#checking cacellation dates
print("Cancel before booking:",(data["cancellation_date"] < data["booking_timestamp"].dt.tz_localize(None)).sum()) #this is outputting 5179
print("Cancel after check in:",(data["cancellation_date"] > data["check_in_date"]).sum())

cancel_before_booking = data[data["cancellation_date"] < data["booking_timestamp"].dt.tz_localize(None)]
print("\nCancel before booking check by Qualification code:")
print(cancel_before_booking["qualification_code"].value_counts())

print("\nQNS with cancellation date:", data[(data["qualification_code"] == "QNS")& (data["cancellation_date"].notna())].shape[0])

print("QXY missing cancellation date:",data[(data["qualification_code"] == "QXY") &(data["cancellation_date"].isna())].shape[0])

print("QXN missing cancellation date:",data[(data["qualification_code"] == "QXN") &(data["cancellation_date"].isna())].shape[0])

#checking if cancellation happened before booking
print("Cancellation date before booking date:",(data["cancellation_date"].dt.date < data["booking_timestamp"].dt.date).sum())
#okay so noting down out of 5179 records that canceled beforebooking
#1186 were same day booking/cancellation and were false positives from the timestamp comparison
#BUT 3993 have a cancellation date that is actually earlier than the booking date (SUS)

print("Member enrollment date after booking date:", (data["member_enrollment_date"].dt.date>data["booking_timestamp"].dt.date).sum())
#99198 records have enrollment after booking, keeping them since this could be valid

print("\nFinal data check beofre rest of the data cleaning")
print("Rows:", len(data))
print("Unique confirmation numbers:", data["confirmation_number"].nunique())
print("Duplicate confirmation numbers:", data["confirmation_number"].duplicated().sum())
print("\nRows per month:")
print(data["month"].value_counts())

#beofre

original_row_count = len(data)

qualification_counts_before = (
    data["qualification_code"]
    .value_counts(dropna=False)
)

rate_code_counts_before = (
    data["rate_code"]
    .value_counts(dropna=False)
)

print("\nORIGINAL ROW COUNT")
print(original_row_count)

print("\nQUALIFICATION CODE COUNTS BEFORE CLEANING")
print(qualification_counts_before)

print("\nRATE CODE COUNTS BEFORE CLEANING")
print(rate_code_counts_before)


#turning IDS to strings

id_columns = [
    "confirmation_number",
    "member_number",
    "site_id"
]

for col in id_columns:
    data[col] = data[col].astype("string")



duplicate_confirmations = data[
    data["confirmation_number"].duplicated(keep=False)
]

print("\nDUPLICATE CONFIRMATION NUMBERS")

if duplicate_confirmations.empty:
    print("None found.")
else:
    print(
        duplicate_confirmations[
            ["confirmation_number", "site_id", "check_in_date", "check_out_date"]
        ].head(10)
    )


#SRB points redeemed

invalid_srb = data[
    (data["rate_code"] == "SRB")
    & (
        data["points_redeemed"].isna()
        | (data["points_redeemed"] == 0)
    )
]

non_srb_with_points = data[
    (data["rate_code"] != "SRB")
    & (data["points_redeemed"].notna())
]

print("\nSRB RECORDS WITH MISSING OR ZERO POINTS_REDEEMED")

if invalid_srb.empty:
    print("None found.")
else:
    print(
        invalid_srb[
            [
                "confirmation_number",
                "rate_code",
                "points_redeemed",
                "site_id",
                "check_in_date"
            ]
        ].head(10)
    )

print("\nNON-SRB RECORDS WITH POINTS_REDEEMED")
print("Count:", len(non_srb_with_points))


#missing points redeeemd rows with 0

missing_points_redeemed_before = data["points_redeemed"].isna().sum()

data["points_redeemed"] = data["points_redeemed"].fillna(0)

print("\nMISSING POINTS_REDEEMED FILLED WITH 0")
print("Rows changed:", missing_points_redeemed_before)


#flagging non memebers but with points earned/redeemed

nonmember_points_issue = data[
    data["member_number"].isna()
    & (data["points_earned"] != 0)
]

print("\nNON-MEMBERS WITH NON-ZERO POINTS_EARNED")

if nonmember_points_issue.empty:
    print("None found.")
else:
    print(
        nonmember_points_issue[
            [
                "confirmation_number",
                "member_number",
                "points_earned",
                "rate_code",
                "site_id"
            ]
        ].head(10)
    )


#Flagging any negative values

numeric_check_columns = [
    "room_revenue",
    "points_earned"
]

for col in numeric_check_columns:

    negative_rows = data[data[col] < 0]

    print(f"\nNEGATIVE VALUES IN {col.upper()}")

    if negative_rows.empty:
        print("None found.")
    else:
        print(
            negative_rows[
                [
                    "confirmation_number",
                    col,
                    "rate_code",
                    "qualification_code",
                    "site_id"
                ]
            ].head(10)
        )

print("\nNEGATIVE VALUES IN POINTS_REDEEMED")
print("Count:", (data["points_redeemed"] < 0).sum())
print("Negative points_redeemed values are expected for redemptions.")


#flagging room_revenue = 0

zero_room_revenue = data[
    data["room_revenue"] == 0
]

print("\nROOM_REVENUE = 0")
print("Count:", len(zero_room_revenue))

if not zero_room_revenue.empty:
    print(
        zero_room_revenue[
            [
                "confirmation_number",
                "room_revenue",
                "rate_code",
                "qualification_code",
                "site_id",
                "check_in_date",
                "check_out_date"
            ]
        ].head(10)
    )


#flagging IQR using outliers

for col in ["room_revenue", "points_earned"]:

    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)

    iqr = q3 - q1

    upper_bound = q3 + (1.5 * iqr)

    high_outliers = data[
        data[col] > upper_bound
    ]

    print(f"\nHIGH OUTLIERS FOR {col.upper()}")
    print("Upper bound:", upper_bound)
    print("Number of flagged rows:", len(high_outliers))

    if not high_outliers.empty:
        print(
            high_outliers[
                [
                    "confirmation_number",
                    col,
                    "rate_code",
                    "qualification_code",
                    "site_id"
                ]
            ].sort_values(
                by=col,
                ascending=False
            ).head(10)
        )


#counts after clean

final_row_count = len(data)

qualification_counts_after = (
    data["qualification_code"]
    .value_counts(dropna=False)
)

rate_code_counts_after = (
    data["rate_code"]
    .value_counts(dropna=False)
)

print("\nFINAL ROW COUNT")
print(final_row_count)

print("\nQUALIFICATION CODE COUNTS AFTER CLEANING")
print(qualification_counts_after)

print("\nRATE CODE COUNTS AFTER CLEANING")
print(rate_code_counts_after)


#verifying if anything is removed

print("\nROW COUNT CHECK")

if original_row_count == final_row_count:
    print("PASS: No records were removed.")
else:
    print(
        "WARNING:",
        original_row_count - final_row_count,
        "records were removed."
    )


print("\nQUALIFICATION CODE COUNT CHECK")

if qualification_counts_before.equals(qualification_counts_after):
    print("PASS: All qualification_code counts stayed the same.")
else:
    print("WARNING: qualification_code counts changed.")


print("\nRATE CODE COUNT CHECK")

if rate_code_counts_before.equals(rate_code_counts_after):
    print("PASS: All rate_code counts stayed the same.")
else:
    print("WARNING: rate_code counts changed.")


print("\nQNS/QXY COUNTS")
print(
    data[data["qualification_code"].isin(["QNS", "QXY"])]
    ["qualification_code"]
    .value_counts()
)

#saving cleaned data

data.to_parquet("data/combined_cleaned.parquet", index=False)

print("\nFinal cleaned data saved.")
