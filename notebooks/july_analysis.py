import pandas as pd
import matplotlib.pyplot as plt

reservations = pd.read_csv("data/reservations_july_2026.csv")
stays = pd.read_csv("data/stays_july_2026.csv")

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
print("\nMISSING VALUES FROM RESERVATIONS")
print(reservations.isnull().sum())

print("\nMISSING VALUES - STAYS")
print(stays.isnull().sum())


#for duplicates
print("\nDUPLICATE CONFIRMATION NUMBERS")
print("Reservations:", reservations["confirmation_number"].duplicated().sum())
print("Stays:", stays["confirmation_number"].duplicated().sum())


#cancel
print("\nCANCELLATIONS")
print("Reservations cancelled:",
      reservations["cancellation_date"].notna().sum())

print("Stays with cancellation date:",
      stays["cancellation_date"].notna().sum())

print("\nMEMBERS")
print("Reservation members:",
      reservations["member_number"].notna().sum())

print("Stay members:",
      stays["member_number"].notna().sum())

print("\nPOINTS REDEEMED")
print(reservations["points_redeemed"].describe())

print("\nPOINTS EARNED")
print(stays["points_earned"].describe())

#check for reservation confirmation numbers are also in stays
matching = reservations["confirmation_number"].isin(
    stays["confirmation_number"]
).sum()

print("\nMATCHING CONFIRMATION NUMBERS")
print("Reservations also found in stays:", matching)
print("Reservations not found in stays:",
      len(reservations) - matching)

print("\nQUALIFICATION CODE COUNTS")
print(stays["qualification_code"].value_counts())

print("\nRATE CODE COUNTS for RESERVATIONS")
print(reservations["rate_code"].value_counts())

print("\nRATE CODE COUNTS for STAYS")
print(stays["rate_code"].value_counts())

print("\nSRB COUNT")
print("Reservation SRB:", (reservations["rate_code"] == "SRB").sum())
print("Stay SRB:", (stays["rate_code"] == "SRB").sum())

print("\nBOOKING CHANNEL COUNTS")
print(reservations["booking_channel"].value_counts())

#Couting unique members (not the non members, those who can gain points)
print("\nUNIQUE MEMBERS")

print("Unique members in reservations:",reservations["member_number"].nunique())

print("Unique members in stays:", stays["member_number"].nunique())

print("\nRESERVATIONS PER MEMBER")
print(reservations["member_number"].value_counts().describe())
print("\nTOP 10 MEMBERS BY RESERVATION COUNT")
print(reservations["member_number"].value_counts().head(10))

#QNS and QXY analysis becuase these are high indicators of fraudulent behavior amongs members
#that has to start with rows where the stay was QNS or QXYonly
qns_qxy = stays[stays["qualification_code"].isin(["QNS", "QXY"])].copy()

print("\nQNS/QXY MEMBER ANALYSIS")

qns_members = stays[stays["qualification_code"] == "QNS"]["member_number"].nunique()
print("Unique members with QNS:", qns_members)

qxy_members = stays[stays["qualification_code"] == "QXY"]["member_number"].nunique()
print("Unique members with QXY:", qxy_members)

print("Unique members with QNS or QXY:",qns_qxy["member_number"].nunique())

#counts for how many QNS/QXY events each member has
member_qns_qxy_counts = (qns_qxy["member_number"].value_counts())

print("\nQNS/QXY EVENTS PER MEMBER")
print(member_qns_qxy_counts.describe())

print("\nTOP 10 MEMBERS BY QNS/QXY COUNT")
print(member_qns_qxy_counts.head(10))

#members with more than 1 QNS/QXY activity
repeated_members = member_qns_qxy_counts[member_qns_qxy_counts > 1]
print("\nMEMBERS WITH MORE THAN 1 QNS/QXY EVENT")
print("Number of members:", len(repeated_members))

print("Highest QNS/QXY count for one member:",member_qns_qxy_counts.max())

print("\nPOINTS EARNED FROM QNS/QXY")
print(qns_qxy["points_earned"].describe())
print("Total points earned from QNS/QXY:",qns_qxy["points_earned"].sum())

# points earned by qualification code
print("\nPOINTS EARNED BY QNS VS QXY")
print(qns_qxy.groupby("qualification_code")["points_earned"].agg(["count", "sum", "mean", "max"]))

print("\nROOM REVENUE - RESERVATIONS")
print(reservations["room_revenue"].describe())

print("\nROOM REVENUE CHECKS")
print("Zero revenue:", (reservations["room_revenue"] == 0).sum())
print("Negative revenue:", (reservations["room_revenue"] < 0).sum())


# EDA visualizations

#for qualification codes
stays["qualification_code"].value_counts().plot(kind="bar")
plt.title("May Qualification Code Counts")
plt.xlabel("Qualification Code")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

reservations["booking_channel"].value_counts().plot(kind="bar")
plt.title("May Reservations by Booking Channel")
plt.xlabel("Booking Channel")
plt.ylabel("Number of Reservations")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

stays["points_earned"].plot(
    kind="hist",
    bins=30
)
plt.title("May Points Earned Distribution")
plt.xlabel("Points Earned")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

reservations["room_revenue"].plot(
    kind="hist",
    bins=30
)
plt.title("May Room Revenue Distribution")
plt.xlabel("Room Revenue")
plt.ylabel("Number of Reservations")
plt.tight_layout()
plt.show()
