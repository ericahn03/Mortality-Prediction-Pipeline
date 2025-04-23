import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load dataset
file_path = "data/IHME_GBD_countrydata.csv"
df = pd.read_csv(file_path, encoding='latin1')

# Clean numeric columns
for col in ['Death Rate Per 100,000', 'Number of Deaths']:
    df[col] = df[col].str.replace(',', '', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

# Convert Age Group to meaningful numeric values (midpoints in years)
age_group_midpoints = {
    '0-6 days': 0.01, '7-27 days': 0.05, '28-364 days': 0.5,
    '1-4 years': 2.5, '5-9 years': 7, '10-14 years': 12,
    '15-19 years': 17, '20-24 years': 22, '25-29 years': 27,
    '30-34 years': 32, '35-39 years': 37, '40-44 years': 42,
    '45-49 years': 47, '50-54 years': 52, '55-59 years': 57,
    '60-64 years': 62, '65-69 years': 67, '70-74 years': 72,
    '75-79 years': 77, '80+ years': 85
}
df["Age (midpoint)"] = df["Age Group"].map(age_group_midpoints)

# Label encode categoricals (for tree models)
label_encoders = {}
categorical_cols = ['Country Name', 'Country Code', 'Sex']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Add log(Number of Deaths) as a feature
df["Log Deaths"] = np.log1p(df["Number of Deaths"])  # log(1 + x) for stability

# Save preprocessed dataset
output_path = "data/preprocessed_data.csv"
df.to_csv(output_path, index=False)
print(f"Preprocessed data saved to: {output_path}")