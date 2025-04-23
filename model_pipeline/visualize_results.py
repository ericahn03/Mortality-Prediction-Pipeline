import pandas as pd
import matplotlib.pyplot as plt

# Load results
results_df = pd.read_csv("data/model_results.csv")

# Ensure run time is numeric
results_df["Run Time (s)"] = pd.to_numeric(results_df["Run Time (s)"], errors='coerce')

# Sort by R² descending by default
results_df = results_df.sort_values(by="R2", ascending=False)

# Define a generic plotting function
def plot_metric(df, metric, color, ylabel, title):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["Model"], df[metric], color=color)
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02 * height,
                 f"{height:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()

# Plot R², MAE, RMSE, Runtime
plot_metric(results_df, "R2", "skyblue", "R² Score", "Model Comparison: R² Scores")
plot_metric(results_df, "MAE", "lightgreen", "Mean Absolute Error", "Model Comparison: MAE")
plot_metric(results_df, "RMSE", "salmon", "Root Mean Squared Error", "Model Comparison: RMSE")
plot_metric(results_df, "Run Time (s)", "gray", "Run Time (seconds)", "Model Comparison: Training Time")
