import pandas as pd

#May
reservations = pd.read_csv("data/revision4/reservations_may_2026.csv")
stays = pd.read_csv("data/revision4/stays_may_2026.csv")

may = reservations.merge(
    stays[["confirmation_number", "points_earned", "qualification_code"]],
    on="confirmation_number",
    how="inner"
)

print("May rows:", len(may))
print("May unique confirmations:", may["confirmation_number"].nunique())
print("May duplicates:", may["confirmation_number"].duplicated().sum())

#June
reservations = pd.read_csv("data/revision4/reservations_june_2026.csv")
stays = pd.read_csv("data/revision4/stays_june_2026.csv")

june = reservations.merge(
    stays[["confirmation_number", "points_earned", "qualification_code"]],
    on="confirmation_number",
    how="inner"
)

print("June rows:", len(june))
print("June unique confirmations:", june["confirmation_number"].nunique())
print("June duplicates:", june["confirmation_number"].duplicated().sum())

#July
reservations = pd.read_csv("data/revision4/reservations_july_2026.csv")
stays = pd.read_csv("data/revision4/stays_july_2026.csv")

july = reservations.merge(
    stays[["confirmation_number", "points_earned", "qualification_code"]],
    on="confirmation_number",
    how="inner"
)

print("July rows:", len(july))
print("July unique confirmations:", july["confirmation_number"].nunique())
print("July duplicates:", july["confirmation_number"].duplicated().sum())

#Adding month columns
may["month"] = "May"
june["month"] = "June"
july["month"] = "July"

#Combining months
combined_months = pd.concat([may, june, july], ignore_index=True)

print("\nCOMBINED MONTHS")
print("Rows:", len(combined_months))
print("Columns:", len(combined_months.columns))
print(combined_months["month"].value_counts())

#SAVEEE
combined_months.to_parquet(
    "data/revision4_combined_may_june_july.parquet",
    index=False
)

print("\nRevision 4 combined data saved.")