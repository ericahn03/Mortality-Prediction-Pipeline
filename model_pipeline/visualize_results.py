import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

# Load model results
base_df = pd.read_csv("data/model_results.csv")
ensemble_df = pd.read_csv("data/ensemble_results.csv")

# Add model type
base_df['Type'] = 'Base'
ensemble_df['Type'] = 'Ensemble'

# Merge both datasets
all_df = pd.concat([base_df, ensemble_df], ignore_index=True)

# Ensure runtime column is numeric
all_df["Run Time (s)"] = pd.to_numeric(all_df["Run Time (s)"], errors='coerce')

# Remove rows with R² = 0 and runtime = 0 (likely placeholder or bug)
all_df = all_df[(all_df["R2"] > 0.001) | (all_df["Run Time (s)"] > 0.001)]

# Sort by R² for cleaner bar charts
all_df = all_df.sort_values(by="R2", ascending=False)

# Plotting function
def plot_grouped_metric(df, metric, ylabel, title):
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x="Model", y=metric, hue="Type", palette="Set2")
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")

    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.02 * height,
                f'{height:.2f}', ha="center", fontsize=9)

    plt.tight_layout()
    plt.show()

# Plot comparisons
plot_grouped_metric(all_df, "R2", "R² Score", "Base vs Ensemble Models: R² Scores")
plot_grouped_metric(all_df, "MAE", "Mean Absolute Error", "Base vs Ensemble Models: MAE")
plot_grouped_metric(all_df, "RMSE", "Root Mean Squared Error", "Base vs Ensemble Models: RMSE")
plot_grouped_metric(all_df, "Run Time (s)", "Seconds", "Base vs Ensemble Models: Training Time")

# ----------------------------
# Feature Importance Plot (Random Forest)
# ----------------------------

# Load full dataset
df = pd.read_csv("data/preprocessed_data.csv")
df.dropna(inplace=True)

feature_cols = ['Country Name', 'Sex', 'Year', 'Age (midpoint)', 'Log Deaths']
X = df[feature_cols]
y = df["Death Rate Per 100,000"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

importances = model.feature_importances_

plt.figure(figsize=(8, 5))
bars = plt.barh(feature_cols, importances, color="steelblue")
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
             f"{width:.2f}", va="center", fontsize=9)

plt.tight_layout()
plt.show()