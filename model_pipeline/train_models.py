import pandas as pd
import numpy as np
import time
import csv

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load preprocessed data
df = pd.read_csv("data/preprocessed_data.csv")
df.dropna(inplace=True)

# Select features and target
feature_cols = ['Country Name', 'Sex', 'Year', 'Age (midpoint)', 'Log Deaths']
X = df[feature_cols]
y = df["Death Rate Per 100,000"]

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(min_samples_leaf=2, random_state=42),
    "Support Vector Regression": make_pipeline(StandardScaler(), LinearSVR(max_iter=5000, random_state=42))
}

# Set up cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize results CSV with Evaluation column
with open("data/model_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "MAE", "RMSE", "R2", "Run Time (s)", "Evaluation"])

# Train & evaluate
for name, model in models.items():
    start = time.time()

    if "80/20 Split" in name:
        # 80/20 Split for SVR
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        if "Support Vector Regression" in name:
            evaluation = "80/20 Split"
        else:
            evaluation = "10-Fold CV"

        print(f"\n{name} ({evaluation}):")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.2f}")

    else:
        # 10-fold CV
        mae_scores = -cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=kf)
        rmse_scores = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf))
        r2_scores = cross_val_score(model, X, y, scoring='r2', cv=kf)

        mae = mae_scores.mean()
        rmse = rmse_scores.mean()
        r2 = r2_scores.mean()
        if "Support Vector Regression" in name:
            evaluation = "80/20 Split"
        else:
            evaluation = "10-Fold CV"

        print(f"\n{name} ({evaluation}):")
        print(f"MAE: {mae:.2f} ± {mae_scores.std():.2f}")
        print(f"RMSE: {rmse:.2f} ± {rmse_scores.std():.2f}")
        print(f"R² Score: {r2:.2f} ± {r2_scores.std():.2f}")

    end = time.time()
    elapsed = end - start

    # Save to CSV
    with open("data/model_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, mae, rmse, r2, round(elapsed, 2), evaluation])