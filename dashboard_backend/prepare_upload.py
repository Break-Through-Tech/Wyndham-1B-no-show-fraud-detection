
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "revision4"

reservations = pd.read_csv(DATA / "reservations_may_2026.csv")
stays = pd.read_csv(DATA / "stays_may_2026.csv")

# Validate reservation IDs before joining
for name, df in [("Reservations", reservations), ("Stays", stays)]:
    if df["confirmation_number"].isna().any():
        raise ValueError(f"{name} contains missing confirmation numbers")
    if df["confirmation_number"].duplicated().any():
        raise ValueError(f"{name} contains duplicate confirmation numbers")

# Add stay outcomes to reservation records
combined = reservations.merge(
    stays[["confirmation_number", "points_earned", "qualification_code"]],
    on="confirmation_number",
    how="inner",
    validate="one_to_one"
)

print("Reservation rows:", len(reservations))
print("Stay rows:", len(stays))
print("Matched rows:", len(combined))
print("Unmatched reservations:", len(reservations) - len(combined))
print("Combined columns:", len(combined.columns))

print("\nUpload preparation test successful!")

