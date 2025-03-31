import matplotlib.pyplot as plt
import pandas as pd

# Load model performance data
results_df = pd.read_csv("data/model_results.csv")

# Convert Run Time column to numeric if it contains strings
if "Run Time (s)" in results_df.columns:
    results_df["Run Time (s)"] = pd.to_numeric(results_df["Run Time (s)"], errors='coerce')

# Plot R^2 Scores
plt.figure(figsize=(10, 6))
plt.bar(results_df["Model"], results_df["R2"], color="skyblue")
plt.title("Model Comparison: R^2 Scores")
plt.ylabel("R^2 Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Plot MAE
plt.figure(figsize=(10, 6))
plt.bar(results_df["Model"], results_df["MAE"], color="lightgreen")
plt.title("Model Comparison: Mean Absolute Error (MAE)")
plt.ylabel("MAE")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Plot RMSE
plt.figure(figsize=(10, 6))
plt.bar(results_df["Model"], results_df["RMSE"], color="salmon")
plt.title("Model Comparison: Root Mean Squared Error (RMSE)")
plt.ylabel("RMSE")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Optional: Plot run time if available
if "Run Time (s)" in results_df.columns and results_df["Run Time (s)"].notnull().all():
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results_df["Model"], results_df["Run Time (s)"], color="gray")
    plt.title("Model Comparison: Run Time")
    plt.ylabel("Seconds")
    plt.xticks(rotation=45, ha="right")

    # Add labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
                 f"{height:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()
