from sklearn.base import clone
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Load lapse dataset
df = pd.read_csv(
    BASE_DIR / "outputs" / "lapse_dataset.csv"
)

# Separate target and features
y = df["lapsed"]

X = df.drop(columns=["customer_id", "lapsed"])

# Identify feature types
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("TRAINING ROWS:", len(X_train))
print("TESTING ROWS:", len(X_test))

# Logistic Regression pipeline
logistic_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

logistic_parameters = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

logistic_search = GridSearchCV(
    logistic_pipeline,
    logistic_parameters,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\nTraining Logistic Regression...")
logistic_search.fit(X_train, y_train)

# Random Forest pipeline
forest_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

forest_parameters = {
    "model__max_depth": [3, 5, 8],
    "model__min_samples_leaf": [5, 10, 20]
}

forest_search = GridSearchCV(
    forest_pipeline,
    forest_parameters,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\nTraining Random Forest...")
forest_search.fit(X_train, y_train)

# Evaluate models on untouched test data
def evaluate_model(name, model):
    predictions = model.predict(X_test)

    print(f"\n{name} RESULTS")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("Precision:", round(
        precision_score(y_test, predictions, zero_division=0), 3
    ))
    print("Recall:", round(
        recall_score(y_test, predictions, zero_division=0), 3
    ))
    print("F1 Score:", round(
        f1_score(y_test, predictions, zero_division=0), 3
    ))

print("\nBEST PARAMETERS:")
print("Logistic Regression:", logistic_search.best_params_)
print("Random Forest:", forest_search.best_params_)

evaluate_model("LOGISTIC REGRESSION", logistic_search.best_estimator_)
evaluate_model("RANDOM FOREST", forest_search.best_estimator_)


# Balance the training data by oversampling lapsed customers
training_data = X_train.copy()
training_data["lapsed"] = y_train.values

majority_count = training_data["lapsed"].value_counts().max()

majority_class = training_data[
    training_data["lapsed"] == 0
]

minority_class = training_data[
    training_data["lapsed"] == 1
]

minority_oversampled = minority_class.sample(
    n=majority_count,
    replace=True,
    random_state=42
)

balanced_data = pd.concat(
    [majority_class, minority_oversampled]
).sample(frac=1, random_state=42)

X_balanced = balanced_data.drop(columns=["lapsed"])
y_balanced = balanced_data["lapsed"]

print("\nBALANCED TRAINING DATA:")
print(y_balanced.value_counts())

# Refit Logistic Regression on balanced training data
balanced_logistic = clone(
    logistic_search.best_estimator_
)

balanced_logistic.fit(X_balanced, y_balanced)

# Evaluate on the original, unchanged test data
balanced_predictions = balanced_logistic.predict(X_test)

print("\nBALANCED LOGISTIC REGRESSION RESULTS")
print("Confusion Matrix:")
print(confusion_matrix(y_test, balanced_predictions))

print("Precision:", round(
    precision_score(y_test, balanced_predictions), 3
))
print("Recall:", round(
    recall_score(y_test, balanced_predictions), 3
))
print("F1 Score:", round(
    f1_score(y_test, balanced_predictions), 3
))

# Save the balanced Logistic Regression model
import joblib

model_path = BASE_DIR / "outputs" / "lapse_model.joblib"

joblib.dump(balanced_logistic, model_path)

print("\nBEST MODEL SAVED:", model_path)
print("\nMODEL TRAINING COMPLETED")
