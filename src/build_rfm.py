
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

orders = pd.read_csv(
    BASE_DIR / "data" / "clean" / "orders.csv"
)

# Convert order timestamps to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Keep only completed orders
completed = orders[orders["status"] == "completed"].copy()

# Snapshot date
snapshot_date = pd.Timestamp("2025-10-01 00:00:00")

# Calculate RFM values
rfm = completed.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

# Calculate recency in whole days
rfm["recency"] = (
    snapshot_date - rfm["last_order"]
).dt.days

# Remove the temporary last_order column
rfm = rfm.drop(columns=["last_order"])

# Arrange columns
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

# Save RFM results
output_path = BASE_DIR / "outputs" / "rfm.csv"
rfm.to_csv(output_path, index=False)

# Display results
print("RFM ANALYSIS COMPLETED")
print("Total customers:", len(rfm))

print("\nRFM SUMMARY:")
print(rfm[["recency", "frequency", "monetary"]].describe())

print("\nREFERENCE CHECKS:")
print("Recency median:", rfm["recency"].median())
print("Minimum recency:", rfm["recency"].min())
print("Maximum recency:", rfm["recency"].max())
print("Mean frequency:", round(rfm["frequency"].mean(), 2))
print("Maximum frequency:", rfm["frequency"].max())
print("Mean monetary:", round(rfm["monetary"].mean(), 2))
print("Maximum monetary:", rfm["monetary"].max())

print("\nSaved:", output_path)
