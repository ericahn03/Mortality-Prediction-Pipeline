import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed dataset
file_path = "data/preprocessed_data.csv"
df = pd.read_csv(file_path)

# Plot distribution of death rate (original and log)
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['Death Rate Per 100,000'], bins=50, kde=True, color='coral')
plt.title('Original Death Rate Distribution')
plt.xlabel('Death Rate per 100k')

plt.subplot(1, 2, 2)
sns.histplot(np.log10(df['Death Rate Per 100,000']), bins=50, kde=True, color='royalblue')
plt.title('Log-Transformed Death Rate Distribution')
plt.xlabel('Log10(Death Rate per 100k)')

plt.tight_layout()
plt.show()

# Correlation matrix for numeric features
plt.figure(figsize=(10, 8))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Scatter plot: Age vs Death Rate
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Age (midpoint)', y='Death Rate Per 100,000', data=df, alpha=0.5)
plt.title("Age vs Death Rate")
plt.xlabel("Age (midpoint)")
plt.ylabel("Death Rate per 100k")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# Scatter plot: Year vs Death Rate
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Year', y='Death Rate Per 100,000', data=df, alpha=0.5)
plt.title("Year vs Death Rate")
plt.xlabel("Year")
plt.ylabel("Death Rate per 100k")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

print("Available columns:", df.columns.tolist())