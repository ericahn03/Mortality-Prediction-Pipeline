import pandas as pd
import numpy as np
import time
import csv

from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load preprocessed data
df = pd.read_csv("data/preprocessed_data.csv")
df.dropna(inplace=True)

# Define features and target
feature_cols = ['Country Name', 'Sex', 'Year', 'Age (midpoint)', 'Log Deaths']
X = df[feature_cols]
y = df["Death Rate Per 100,000"]

# Define 10-fold CV
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Define ensemble models
models = {
    "Random Forest (10-Fold CV)": RandomForestRegressor(n_estimators=100, random_state=42),
    "Bagging (Tree) (10-Fold CV)": BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=10, random_state=42),
    "Stacking (Tree + Linear → Ridge) (10-Fold CV)": StackingRegressor(
        estimators=[
            ("tree", DecisionTreeRegressor(max_depth=5, random_state=42)),
            ("linear", make_pipeline(StandardScaler(), LinearRegression()))
        ],
        final_estimator=make_pipeline(StandardScaler(), RidgeCV())
    )
}

# Initialize CSV with Evaluation column
with open("data/ensemble_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "MAE", "RMSE", "R2", "Run Time (s)", "Evaluation"])

# Evaluate models with CV
for name, model in models.items():
    start = time.time()

    mae_scores = -cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=kf)
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf))
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=kf)

    end = time.time()
    elapsed = end - start

    print(f"\n{name} (10-fold CV):")
    print(f"MAE: {mae_scores.mean():.2f} ± {mae_scores.std():.2f}")
    print(f"RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
    print(f"R² Score: {r2_scores.mean():.2f} ± {r2_scores.std():.2f}")
    print(f"Run Time: {elapsed:.2f} sec")

    with open("data/ensemble_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, mae_scores.mean(), rmse_scores.mean(), r2_scores.mean(), round(elapsed, 2), "10-Fold CV"])