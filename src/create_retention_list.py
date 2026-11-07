
from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

# Load customer data and trained model
segments = pd.read_csv(
    BASE_DIR / "outputs" / "customer_segments.csv"
)

lapse_data = pd.read_csv(
    BASE_DIR / "outputs" / "lapse_dataset.csv"
)

model = joblib.load(
    BASE_DIR / "outputs" / "lapse_model.joblib"
)

# Select Premium customers
premium_customers = segments[
    segments["is_premium"] == True
].copy()

# Prepare model features
feature_columns = [
    column for column in lapse_data.columns
    if column not in ["customer_id", "lapsed"]
]

# Match Premium customers with their lapse features
premium_data = premium_customers[
    ["customer_id"]
].merge(
    lapse_data,
    on="customer_id",
    how="inner"
)

X_premium = premium_data[feature_columns]

# Predict probability of lapsing
premium_data["lapse_probability"] = (
    model.predict_proba(X_premium)[:, 1]
)

# Include actual lapse status for evaluation
premium_data["actually_lapsed"] = premium_data["lapsed"]

# Select the 10 highest-risk Premium customers
retention_list = premium_data.sort_values(
    "lapse_probability",
    ascending=False
).head(10)

# Keep required columns
retention_list = retention_list[
    [
        "customer_id",
        "lapse_probability",
        "actually_lapsed"
    ]
]

# Save retention list
output_path = BASE_DIR / "outputs" / "retention_list.csv"

retention_list.to_csv(
    output_path,
    index=False
)

print("RETENTION LIST CREATED")

print("\nTOP 10 PREMIUM CUSTOMERS:")
print(retention_list.to_string(index=False))

print(
    "\nActually lapsed customers in top 10:",
    int(retention_list["actually_lapsed"].sum())
)

print("\nSaved:", output_path)
