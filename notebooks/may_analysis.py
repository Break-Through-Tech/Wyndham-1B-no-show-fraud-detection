import pandas as pd

reservations = pd.read_csv("data/reservations_may_2026.csv")

stays = pd.read_csv("data/stays_may_2026.csv")

print("RESERVATIONS")
print("Shape:", reservations.shape)
print(reservations.head())
print("\nColumns:")
print(reservations.columns.tolist())

print("\nSTAYS")
print("Shape:", stays.shape)
print(stays.head())
print("\nColumns:")
print(stays.columns.tolist())


#data types check
print("\nRESERVATION DATA TYPES")
print(reservations.dtypes)

print("\nSTAY DATA TYPES")
print(stays.dtypes)


#missing data
print("\nMISSING VALUES - RESERVATIONS")
print(reservations.isnull().sum())

print("\nMISSING VALUES - STAYS")
print(stays.isnull().sum())


#repeating members/numbers checks
print("\nDUPLICATE CONFIRMATION NUMBERS")
print("Reservations:", reservations["confirmation_number"].duplicated().sum())
print("Stays:", stays["confirmation_number"].duplicated().sum())


#cancel
print("\nCANCELLATIONS")
print("Reservations cancelled:",
      reservations["cancellation_date"].notna().sum())

print("Stays with cancellation date:",
      stays["cancellation_date"].notna().sum())


#membership
print("\nMEMBERS")
print("Reservation members:",
      reservations["member_number"].notna().sum())

print("Stay members:",
      stays["member_number"].notna().sum())


# points
print("\nPOINTS REDEEMED")
print(reservations["points_redeemed"].describe())

print("\nPOINTS EARNED")
print(stays["points_earned"].describe())

#checking if reservation confirmation numbers are also in stays
matching = reservations["confirmation_number"].isin(
    stays["confirmation_number"]
).sum()

print("\nMATCHING CONFIRMIRMATION NUMBERS")
print("Reservations also found in stays:", matching)
print("Reservations not found in stays:",
      len(reservations) - matching)