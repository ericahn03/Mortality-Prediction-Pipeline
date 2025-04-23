import pandas as pd
import numpy as np
import time
import csv

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load preprocessed data
df = pd.read_csv("data/preprocessed_data.csv")

df.dropna(inplace=True)

# Select useful features explicitly
feature_cols = ['Country Name', 'Sex', 'Year', 'Age (midpoint)', 'Log Deaths']
X = df[feature_cols]
y = df["Death Rate Per 100,000"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models with consistent pipelines
models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Support Vector Regression": make_pipeline(StandardScaler(), SVR())
}

# Initialize CSV file
with open("data/model_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "MAE", "RMSE", "R2", "Run Time (s)"])

# Train and evaluate each model
for name, model in models.items():
    start_time = time.time()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    end_time = time.time()
    elapsed = end_time - start_time

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name} Results:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    print(f"Run Time: {elapsed:.2f} seconds")

    # Save to CSV
    with open("data/model_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, mae, rmse, r2, round(elapsed, 2)])
