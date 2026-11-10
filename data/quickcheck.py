import pandas as pd

data = pd.read_parquet("data/revision4_combined_cleaned.parquet")

print(data.columns.tolist())