import pandas as pd

file_path = "/Users/svalisammagari/Downloads/combined_may_june_july.parquet"

data = pd.read_parquet(file_path)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print("Original shape:", data.shape)

print("\nFIRST 5 RECORDS")
print(data.head())

import pandas as pd
import numpy as np

# --------------------------------------------------
# BEFORE CLEANING COUNTS
# --------------------------------------------------

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


# --------------------------------------------------
# 1. MAKE IDS STRINGS
# --------------------------------------------------

id_columns = [
    "confirmation_number",
    "member_number",
    "site_id"
]

for col in id_columns:
    data[col] = data[col].astype("string")


# --------------------------------------------------
# 2. CHECK CONFIRMATION NUMBER UNIQUENESS
# --------------------------------------------------

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
        ]
    )


# --------------------------------------------------
# 3. CHECK SRB POINTS REDEEMED
# --------------------------------------------------

invalid_srb = data[
    (data["rate_code"] == "SRB")
    & (
        data["points_redeemed"].isna()
        | (data["points_redeemed"] == 0)
    )
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
        ]
    )


# --------------------------------------------------
# 4. FILL MISSING POINTS_REDEEMED WITH 0
# --------------------------------------------------

data["points_redeemed"] = data["points_redeemed"] * -1

missing_points_redeemed_before = data["points_redeemed"].isna().sum()

data["points_redeemed"] = data["points_redeemed"].fillna(0)

print("\nMISSING POINTS_REDEEMED FILLED WITH 0")
print("Rows changed:", missing_points_redeemed_before)


# --------------------------------------------------
# 5. FLAG NON-MEMBERS WITH POINTS_EARNED != 0
# --------------------------------------------------

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
        ]
    )


# --------------------------------------------------
# 6. FLAG NEGATIVE VALUES
# --------------------------------------------------

numeric_check_columns = [
    "room_revenue",
    "points_earned",
    "points_redeemed"
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
            ]
        )


# --------------------------------------------------
# 7. FLAG ROOM_REVENUE = 0
# --------------------------------------------------

zero_room_revenue = data[
    data["room_revenue"] == 0
]

print("\nROOM_REVENUE = 0")

if zero_room_revenue.empty:
    print("None found.")
else:
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
        ]
    )


# --------------------------------------------------
# 8. FLAG HIGH OUTLIERS USING IQR
# --------------------------------------------------

for col in numeric_check_columns:

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
            )
        )


# --------------------------------------------------
# 9. AFTER CLEANING COUNTS
# --------------------------------------------------

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


# --------------------------------------------------
# 10. VERIFY NOTHING WAS REMOVED
# --------------------------------------------------

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