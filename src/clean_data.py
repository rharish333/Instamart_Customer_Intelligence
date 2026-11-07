
from pathlib import Path
import pandas as pd

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "clean"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. LOAD RAW DATA
# ==========================================

customers = pd.read_csv(RAW_DIR / "customers.csv")
products = pd.read_csv(RAW_DIR / "products.csv")
orders = pd.read_csv(RAW_DIR / "orders.csv")

print("Raw data loaded successfully.")

# ==========================================
# 2. CLEAN CUSTOMERS
# ==========================================

# Standardise city names
customers["city"] = (
    customers["city"].astype("string").str.strip().str.title()
)

# Standardise gender values
gender_map = {
    "female": "female",
    "f": "female",
    "male": "male",
    "m": "male"
}

customers["gender"] = (
    customers["gender"]
    .astype("string")
    .str.strip()
    .str.lower()
    .map(gender_map)
    .fillna("unknown")
)

# Convert invalid ages into missing values
customers["age"] = pd.to_numeric(
    customers["age"], errors="coerce"
)

customers.loc[
    ~customers["age"].between(10, 100), "age"
] = float("nan")

# Fill missing ages using the median of each city
city_medians = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_medians)

# Convert signup date
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

print("Customers cleaned.")

# ==========================================
# 3. CLEAN PRODUCTS
# ==========================================

products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="coerce"
)

print("Products cleaned.")

# ==========================================
# 4. CLEAN ORDERS
# ==========================================

# Convert order timestamp
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

# Remove exact duplicate rows
orders = orders.drop_duplicates()

# Keep the latest record for each order ID
orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates(
    subset="order_id", keep="last"
)

# Convert order amount into numeric values
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Get product prices
orders = orders.merge(
    products[["product_id", "price_inr"]],
    on="product_id",
    how="left"
)

# Calculate missing order amounts
calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount
)

# Remove temporary product price column
orders = orders.drop(columns=["price_inr"])

# Convert negative delivery times to missing
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = float("nan")

# Fill missing delivery times only for completed orders
completed_mask = orders["status"].eq("completed")

category_lookup = products.set_index("product_id")["category"]
orders["category"] = orders["product_id"].map(category_lookup)

completed_medians = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed_mask, "delivery_minutes"] = (
    orders.loc[completed_mask, "delivery_minutes"]
    .fillna(completed_medians)
)

# Remove temporary category column
orders = orders.drop(columns=["category"])

print("Orders cleaned.")

# ==========================================
# 5. SAVE CLEANED DATA
# ==========================================

customers.to_csv(
    CLEAN_DIR / "customers.csv", index=False
)

products.to_csv(
    CLEAN_DIR / "products.csv", index=False
)

orders.to_csv(
    CLEAN_DIR / "orders.csv", index=False
)

print("\n=================================")
print("DATA CLEANING COMPLETED")
print("=================================")

print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))

print("\nOrder status counts:")
print(orders["status"].value_counts())

print("\nCompleted orders with missing delivery time:",
      orders.loc[
          orders["status"] == "completed",
          "delivery_minutes"
      ].isna().sum())

print("\nCancelled orders with missing delivery time:",
      orders.loc[
          orders["status"] == "cancelled",
          "delivery_minutes"
      ].isna().sum())

print("\nClean files saved in:", CLEAN_DIR)
print("\nCUSTOMER CHECKPOINTS")
print("\nCity counts:")
print(customers["city"].value_counts())

print("\nGender counts:")
print(customers["gender"].value_counts())

print("\nMissing ages:", customers["age"].isna().sum())
print("Minimum age:", customers["age"].min())
print("Maximum age:", customers["age"].max())

print("\nORDER AMOUNT CHECKPOINTS")
print("Missing order amounts:", orders["order_amount"].isna().sum())

completed_total = orders.loc[
    orders["status"] == "completed",
    "order_amount"
].sum()

print("Completed order amount total:", round(completed_total, 2))