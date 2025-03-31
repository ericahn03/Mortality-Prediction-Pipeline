import pandas as pd

# Load your dataset
df = pd.read_csv('IHME_GBD_countrydata.csv')

# Save dataset as ARFF file
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder

# Encode categorical columns
for column in df.select_dtypes(include=['object']).columns:
    df[column] = LabelEncoder().fit_transform(df[column])

# Save as ARFF
with open('IHME_GBD_2010_MORTALITY.arff', 'w') as f:
    f.write(f'@RELATION mortality\n\n')
    
    for col in df.columns:
        if df[col].dtype == 'int64' or df[col].dtype == 'float64':
            f.write(f'@ATTRIBUTE {col} NUMERIC\n')
        else:
            unique_vals = ','.join([str(i) for i in df[col].unique()])
            f.write(f'@ATTRIBUTE {col} {{{unique_vals}}}\n')
    
    f.write('\n@DATA\n')
    for index, row in df.iterrows():
        f.write(','.join(map(str, row.values)) + '\n')
