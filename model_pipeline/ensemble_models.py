import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load dataset
df = pd.read_csv("data/preprocessed_data.csv")
df.dropna(inplace=True)

if df.isnull().sum().sum() > 0:
    print("WARNING: Data still contains missing values after initial preprocessing.")
    print(df.isnull().sum())

# Select useful features
feature_cols = [
    'Country Name', 'Sex', 'Year', 'Age (midpoint)',
    'Log Deaths'  # Optional: test model both with and without this
]
X = df[feature_cols]
y = df['Death Rate Per 100,000']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Ensemble Models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Bagging (Tree)": BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=50, random_state=42),
    "Stacking (Tree + Linear → Ridge)": StackingRegressor(
        estimators=[
            ("tree", DecisionTreeRegressor(max_depth=5, random_state=42)),
            ("linear", make_pipeline(StandardScaler(), LinearRegression()))
        ],
        final_estimator=make_pipeline(StandardScaler(), RidgeCV())
    )
}

# Evaluate each model
for name, model in models.items():
    print(f"\n{name} Results:")
    start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end = time.time()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"Run Time: {end - start:.2f} sec")
