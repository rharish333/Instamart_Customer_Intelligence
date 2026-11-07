
from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

customers = pd.read_csv(BASE_DIR / "data" / "clean" / "customers.csv")
products = pd.read_csv(BASE_DIR / "data" / "clean" / "products.csv")
orders = pd.read_csv(BASE_DIR / "data" / "clean" / "orders.csv")

# Check 1: Row counts
assert len(customers) == 1200
assert len(products) == 40
assert len(orders) == 12309

# Check 2: Unique order IDs
assert orders["order_id"].nunique() == 12309

# Check 3: Order statuses
assert (orders["status"] == "completed").sum() == 11110
assert (orders["status"] == "cancelled").sum() == 1199

# Check 4: No missing customer ages
assert customers["age"].isna().sum() == 0

# Check 5: No missing order amounts
assert orders["order_amount"].isna().sum() == 0

# Check 6: Completed orders have delivery times
completed = orders[orders["status"] == "completed"]
assert completed["delivery_minutes"].isna().sum() == 0

# Check 7: Cancelled orders have missing delivery times
cancelled = orders[orders["status"] == "cancelled"]
assert cancelled["delivery_minutes"].isna().sum() == 1199

# Check 8: Total completed order revenue
total_revenue = round(completed["order_amount"].sum(), 2)
assert total_revenue == 2998668.80

# Check 9: Database tables and row counts
db_path = BASE_DIR / "data" / "instamart.db"

with sqlite3.connect(db_path) as connection:
    for table, expected in [
        ("customers", 1200),
        ("products", 40),
        ("orders", 12309)
    ]:
        actual = pd.read_sql_query(
            f"SELECT COUNT(*) AS total FROM {table}",
            connection
        )["total"][0]

        assert actual == expected

print("ALL CHECKPOINTS PASSED")
print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))
print("Completed order revenue: INR", total_revenue)
