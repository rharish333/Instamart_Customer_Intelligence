
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

# Load cleaned data
customers = pd.read_csv(BASE_DIR / "data" / "clean" / "customers.csv")
orders = pd.read_csv(BASE_DIR / "data" / "clean" / "orders.csv")
products = pd.read_csv(BASE_DIR / "data" / "clean" / "products.csv")

# Convert dates
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# Define the prediction cutoff and label period
cutoff = pd.Timestamp("2025-09-01")
period_end = pd.Timestamp("2025-10-01")

# Only use orders before September for features
past_orders = orders[orders["order_ts"] < cutoff].copy()

# Completed orders before September
past_completed = past_orders[
    past_orders["status"] == "completed"
].copy()

# Cohort: customers with at least one previous completed order
cohort_ids = past_completed["customer_id"].unique()

# Completed orders during September, used only for the target label
september_completed = orders[
    (orders["order_ts"] >= cutoff)
    & (orders["order_ts"] < period_end)
    & (orders["status"] == "completed")
]

active_ids = set(september_completed["customer_id"])

# Build target labels
dataset = pd.DataFrame({"customer_id": cohort_ids})

dataset["lapsed"] = (
    ~dataset["customer_id"].isin(active_ids)
).astype(int)

# Calculate RFM-style features using past completed orders only
features = past_completed.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery_minutes=("delivery_minutes", "mean")
).reset_index()

features["recency"] = (
    cutoff - features["last_order"]
).dt.days

features = features.drop(columns=["last_order"])

# Calculate cancellation rate using all orders before cutoff
order_counts = past_orders.groupby("customer_id").agg(
    total_orders=("order_id", "nunique"),
    cancelled_orders=(
        "status",
        lambda values: (values == "cancelled").sum()
    )
).reset_index()

order_counts["cancellation_rate"] = (
    order_counts["cancelled_orders"]
    / order_counts["total_orders"]
)

# Find each customer's most frequently purchased category
past_with_products = past_completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

category_features = (
    past_with_products.groupby("customer_id")["category"]
    .agg(lambda values: values.mode().iloc[0])
    .reset_index(name="most_frequent_category")
)

# Add customer information
customer_features = customers[
    ["customer_id", "city", "age", "gender", "signup_date"]
].copy()

customer_features["tenure_days"] = (
    cutoff - customer_features["signup_date"]
).dt.days

customer_features = customer_features.drop(
    columns=["signup_date"]
)

# Combine all features
dataset = dataset.merge(features, on="customer_id", how="left")
dataset = dataset.merge(order_counts, on="customer_id", how="left")
dataset = dataset.merge(
    category_features, on="customer_id", how="left"
)
dataset = dataset.merge(
    customer_features, on="customer_id", how="left"
)

# Remove columns not needed for prediction
dataset = dataset.drop(
    columns=["total_orders", "cancelled_orders"]
)

# Check missing values
print("MISSING VALUES:")
print(dataset.isnull().sum())

# Save dataset
output_path = BASE_DIR / "outputs" / "lapse_dataset.csv"
dataset.to_csv(output_path, index=False)

# Display reference checks
print("\nLAPSE DATASET CREATED")
print("Total customers:", len(dataset))
print("Lapsed customers:", int(dataset["lapsed"].sum()))
print("Active customers:", int((dataset["lapsed"] == 0).sum()))
print(
    "Lapse percentage:",
    round(dataset["lapsed"].mean() * 100, 2), "%"
)

print("\nSaved:", output_path)
