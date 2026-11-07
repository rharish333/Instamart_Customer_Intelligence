
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset
df = pd.read_csv(
    BASE_DIR / "outputs" / "lapse_dataset.csv"
)

X = df.drop(columns=["customer_id", "lapsed"])
y = df["lapsed"]

# Use the same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Load saved model
model = joblib.load(
    BASE_DIR / "outputs" / "lapse_model.joblib"
)

# Verify saved model predictions
predictions_before = model.predict(X_test)

# Save and reload the model
temporary_path = BASE_DIR / "outputs" / "model_check.joblib"

joblib.dump(model, temporary_path)
reloaded_model = joblib.load(temporary_path)

predictions_after = reloaded_model.predict(X_test)

print("MODEL VALIDATION")
print(
    "Predictions identical:",
    np.array_equal(predictions_before, predictions_after)
)

print(
    "Test accuracy:",
    round(accuracy_score(y_test, predictions_after), 3)
)

# Remove temporary validation file
temporary_path.unlink()

# Random Forest feature importance
print("\nRANDOM FOREST FEATURE IMPORTANCE")

# Retrain the best Random Forest using the original training data
from sklearn.base import clone

# This pipeline is not the saved model, so use a separate
# Random Forest with the same best parameters.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["str", "object"]
).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            drop="first",
            handle_unknown="ignore"
        ))
    ]), categorical_features)
])

forest = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42
    ))
])

forest.fit(X_train, y_train)

feature_names = forest.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = forest.named_steps[
    "model"
].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
}).sort_values(
    "importance",
    ascending=False
)

print(importance_df.head(5).to_string(index=False))

importance_df.to_csv(
    BASE_DIR / "outputs" / "feature_importance.csv",
    index=False
)

print("\nSaved: outputs/feature_importance.csv")

print("\nMODEL ANALYSIS COMPLETED")
