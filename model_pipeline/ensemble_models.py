import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import time  # ⏱ Add this for timing

# Load preprocessed data
df = pd.read_csv("data/preprocessed_data.csv")

# Define features and target
X = df.drop(columns=["Death Rate Per 100,000"])
y = df["Death Rate Per 100,000"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define ensemble models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Bagging (Decision Tree)": BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=50, random_state=42),
    "Stacking (Tree + Linear -> LR)": StackingRegressor(
        estimators=[
            ("tree", DecisionTreeRegressor(max_depth=5, random_state=42)),
            ("linear", LinearRegression())
        ],
        final_estimator=LinearRegression()
    )
}

# Train and evaluate each ensemble model
for name, model in models.items():
    start_time = time.time()  # Start timer

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    end_time = time.time()  # End timer
    elapsed = end_time - start_time

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name} Results:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    print(f"Run Time: {elapsed:.2f} seconds")
