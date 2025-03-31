import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "data/IHME_GBD_countrydata.csv"
df = pd.read_csv(file_path, encoding='latin1')

# Clean and convert death rate to float
df['Death Rate Per 100,000'] = df['Death Rate Per 100,000'].str.replace(',', '', regex=False).astype(float)

# Filter out 0 or negative values (log can't handle these)
filtered = df[df['Death Rate Per 100,000'] > 0].copy()

# Create a new column with log10 of death rate
filtered['log_death_rate'] = np.log10(filtered['Death Rate Per 100,000'])

# Plot log-transformed histogram
plt.figure(figsize=(12, 6))
sns.histplot(filtered['log_death_rate'], bins=50, kde=True, color='royalblue')

plt.title('Log-Transformed Distribution of Age-Specific Death Rates', fontsize=14)
plt.xlabel('Log10(Death Rate per 100k)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
