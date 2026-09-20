import pandas as pd 
 
data = pd.read_parquet( 
    "data/revision4_qns_qxy_member_analysis.parquet" 
) 
 
fraud = data[data["is_fraud_account"] == 1] 
legit = data[data["is_fraud_account"] == 0] 
 
print("FRAUD - QNS counts") 
print(fraud["qns_count"].value_counts().sort_index()) 
 
print("\nFRAUD - QXY counts") 
print(fraud["qxy_count"].value_counts().sort_index()) 
 
print("\nFRAUD - QNS/QXY counts") 
print(fraud["qns_qxy_count"].value_counts().sort_index()) 
 
print("\nFRAUD - months with QNS/QXY") 
print(fraud["months_with_qns_qxy"].value_counts().sort_index()) 
 
print("\nLEGIT - QNS/QXY counts") 
print( 
    legit["qns_qxy_count"] 
    .value_counts() 
    .sort_index() 
) 
 
#QNS/QXY rates 
print("\nFRAUD - QNS/QXY rate") 
print(fraud["qns_qxy_rate"].describe()) 
 
print("\nLEGIT - QNS/QXY rate") 
print(legit["qns_qxy_rate"].describe()) 
 
print("\nLEGIT members with 6+ QNS/QXY - rate") 
print( 
    legit[legit["qns_qxy_count"] >= 6] 
    ["qns_qxy_rate"] 
    .describe() 
) 
 
#checking months with QNS/QXY for legit members 
print("\nLEGIT - months with QNS/QXY") 
print( 
    legit["months_with_qns_qxy"] 
    .value_counts() 
    .sort_index() 
) 
 
print("\nFRAUD - months with QNS/QXY") 
print( 
    fraud["months_with_qns_qxy"] 
    .value_counts() 
    .sort_index() 
) 
 
#checking how many fraud and legit members have repeated QNS/QXY 
print("\nMembers with 2+ QNS/QXY") 
print( 
    data[data["qns_qxy_count"] >= 2] 
    ["is_fraud_account"] 
    .value_counts() 
) 
 
print("\nMembers with 6+ QNS/QXY") 
print( 
    data[data["qns_qxy_count"] >= 6] 
    ["is_fraud_account"] 
    .value_counts() 
) 
 
#checking QXY specifically 
print("\nMembers with QXY by fraud label") 
print( 
    data[data["qxy_count"] > 0] 
    ["is_fraud_account"] 
    .value_counts() 
) 
 
print("\nFraud members with QXY") 
print((fraud["qxy_count"] > 0).sum()) 
 
print("\nLegit members with QXY") 
print((legit["qxy_count"] > 0).sum()) 
 
#checking QNS/QXY across all 3 months 
print("\nMembers with QNS/QXY in all 3 months") 
print( 
    data[data["months_with_qns_qxy"] == 3] 
    ["is_fraud_account"] 
    .value_counts() 
) 
 
#comparing members with samehigh counts 
for count in [6, 12]: 
    group = data[data["qns_qxy_count"] == count] 
 
    print(f"\nMembers with exactly {count} QNS/QXY events") 
 
    print("Fraud labels:") 
    print(group["is_fraud_account"].value_counts()) 
 
    print("QNS/QXY rate:") 
    print( 
        group.groupby("is_fraud_account") 
        ["qns_qxy_rate"] 
        .describe() 
    ) 
 
    print("Months with QNS/QXY:") 
    print( 
        group.groupby("is_fraud_account") 
        ["months_with_qns_qxy"] 
        .mean() 
    )