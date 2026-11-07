
from pathlib import Path
import sqlite3
import pandas as pd

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"
DB_PATH = BASE_DIR / "data" / "instamart.db"

# Load cleaned CSV files
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

# Store dates in the required text format
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"]
).dt.strftime("%Y-%m-%d")

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"]
).dt.strftime("%Y-%m-%d %H:%M:%S")

# Create SQLite database and tables
with sqlite3.connect(DB_PATH) as connection:
    customers.to_sql(
        "customers", connection, if_exists="replace", index=False
    )

    products.to_sql(
        "products", connection, if_exists="replace", index=False
    )

    orders.to_sql(
        "orders", connection, if_exists="replace", index=False
    )

    # Verify table row counts
    print("DATABASE CREATED SUCCESSFULLY")

    for table in ["customers", "products", "orders"]:
        count = pd.read_sql_query(
            f"SELECT COUNT(*) AS total FROM {table}",
            connection
        )["total"][0]

        print(f"{table}: {count} rows")

print("\nDatabase saved at:", DB_PATH)
