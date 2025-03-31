import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = "data/IHME_GBD_countrydata.csv"
df = pd.read_csv(file_path, encoding='latin1')

# Clean and convert numeric columns
for col in ['Death Rate Per 100,000', 'Number of Deaths']:
    df[col] = df[col].str.replace(',', '', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to float, handle bad values

# Drop rows with missing values after conversion
df.dropna(inplace=True)

# Encode categorical features
label_encoders = {}
categorical_cols = ['Country Name', 'Country Code', 'Age Group', 'Sex']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save preprocessed data to a new CSV
output_path = "data/preprocessed_data.csv"
df.to_csv(output_path, index=False)
print(f"Preprocessed data saved to: {output_path}")
