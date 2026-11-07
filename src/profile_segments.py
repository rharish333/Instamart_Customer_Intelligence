
from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).resolve().parent.parent

# Load customer clusters
df = pd.read_csv(
    BASE_DIR / "outputs" / "customer_clusters.csv"
)

# Give each segment a business-friendly name
segment_names = {
    0: "Occasional Customers",
    1: "High-Value Customers",
    2: "Frequent Customers",
    3: "At-Risk Customers"
}

df["segment_name"] = df["segment_id"].map(segment_names)

# Identify the segment with the highest average spending
profile = df.groupby("segment_id")["monetary"].mean()
premium_segment = profile.idxmax()

# Mark Premium customers
df["is_premium"] = (
    df["segment_id"] == premium_segment
)

# Save final customer segments
final_columns = [
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "segment_id",
    "segment_name",
    "is_premium"
]

df[final_columns].to_csv(
    BASE_DIR / "outputs" / "customer_segments.csv",
    index=False
)

# Perform PCA using standardized RFM values
features = ["recency", "frequency", "monetary"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

pca = PCA(n_components=2)
coordinates = pca.fit_transform(X_scaled)

# Save PCA coordinates separately
pca_df = pd.DataFrame({
    "customer_id": df["customer_id"],
    "segment_id": df["segment_id"],
    "PC1": coordinates[:, 0],
    "PC2": coordinates[:, 1]
})

pca_df.to_csv(
    BASE_DIR / "outputs" / "pca_coordinates.csv",
    index=False
)

print("CUSTOMER SEGMENT PROFILING COMPLETED")

print("\nSEGMENT NAMES:")
print(
    df.groupby("segment_name").size()
)

print("\nPREMIUM SEGMENT ID:", premium_segment)
print("Premium customers:", int(df["is_premium"].sum()))

print("\nPCA EXPLAINED VARIANCE:")
print(pca.explained_variance_ratio_.round(3))

print(
    "Total variance explained:",
    round(pca.explained_variance_ratio_.sum(), 3)
)

print("\nFILES SAVED:")
print("outputs/customer_segments.csv")
print("outputs/pca_coordinates.csv")
