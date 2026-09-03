import pandas as pd
#May merging

reservations = pd.read_csv("data/reservations_may_2026.csv")
stays = pd.read_csv("data/stays_may_2026.csv")

print("May reservations:", reservations.shape)
print("May stays:", stays.shape)

# Combining stay and reservation data for May based on the confrimations numberwhich is unique ID 
may = reservations.merge(stays[["confirmation_number", "points_earned", "qualification_code"]], on="confirmation_number", how="inner")

print("\nMay after merge")
print("Rows:", len(may))
print("Columns:", len(may.columns))
print("Unique confirmation numbers:", may["confirmation_number"].nunique())
print("Duplicate confirmation numbers:", may["confirmation_number"].duplicated().sum())

print("\nCOLUMNS")
print(may.columns.tolist())

#June Merging
reservations = pd.read_csv("data/reservations_june_2026.csv")
stays = pd.read_csv("data/stays_june_2026.csv")

print("June reservations:", reservations.shape)
print("June stays:", stays.shape)

# Combining stay and reservation data for May based on the confrimations numberwhich is unique ID 
june = reservations.merge(stays[["confirmation_number", "points_earned", "qualification_code"]], on="confirmation_number", how="inner")

print("\nJune after merge")
print("Rows:", len(june))
print("Columns:", len(june.columns))
print("Unique confirmation numbers:", june["confirmation_number"].nunique())
print("Duplicate confirmation numbers:", june["confirmation_number"].duplicated().sum())

print("\nCOLUMNS")
print(june.columns.tolist())

#July
reservations = pd.read_csv("data/reservations_july_2026.csv")
stays = pd.read_csv("data/stays_july_2026.csv")

print("July reservations:", reservations.shape)
print("July stays:", stays.shape)

# Combining stay and reservation data for May based on the confrimations numberwhich is unique ID 
july = reservations.merge(stays[["confirmation_number", "points_earned", "qualification_code"]], on="confirmation_number", how="inner")

print("\nJuly after merge")
print("Rows:", len(july))
print("Columns:", len(july.columns))
print("Unique confirmation numbers:", july["confirmation_number"].nunique())
print("Duplicate confirmation numbers:", july["confirmation_number"].duplicated().sum())

print("\nCOLUMNS")
print(july.columns.tolist())

#im gonna combine all months here
may["month"] = "May"
june["month"] = "June"
july["month"] = "July"

combined_months = pd.concat([may, june, july], ignore_index=True)

print("\nCOMBINED MONTHS DATA")
print("Rows:", len(combined_months))
print("Columns:", len(combined_months.columns))
print(combined_months["month"].value_counts())

#im using parquet to save the huge file (9.6M rows) so it takes less disk space than CSV
combined_months.to_parquet("data/combined_may_june_july.parquet", index=False)
