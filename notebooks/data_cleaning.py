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

#saving cleaned data for the next cleaning step
data.to_parquet("data/combined_person1_clean.parquet", index=False)