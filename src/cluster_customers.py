
from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parent.parent

# Load RFM data
rfm = pd.read_csv(BASE_DIR / "outputs" / "rfm.csv")

# Select RFM features
features = ["recency", "frequency", "monetary"]
X = rfm[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test different numbers of clusters
results = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    results.append({
        "clusters": k,
        "inertia": round(model.inertia_, 2),
        "silhouette_score": round(
            silhouette_score(X_scaled, labels), 3
        )
    })

evaluation = pd.DataFrame(results)

evaluation.to_csv(
    BASE_DIR / "outputs" / "cluster_evaluation.csv",
    index=False
)

print("CLUSTER EVALUATION:")
print(evaluation.to_string(index=False))

# Create final customer segments using 4 clusters
final_model = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = final_model.fit_predict(X_scaled)

# Save customer clusters
rfm.to_csv(
    BASE_DIR / "outputs" / "customer_clusters.csv",
    index=False
)

# Display customer segment profiles
profile = rfm.groupby("segment_id").agg(
    customer_count=("customer_id", "count"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean")
).round(2)

print("\nCUSTOMER SEGMENT PROFILES:")
print(profile)

print("\nCLUSTERING COMPLETED SUCCESSFULLY")
print("Saved: outputs/cluster_evaluation.csv")
print("Saved: outputs/customer_clusters.csv")
