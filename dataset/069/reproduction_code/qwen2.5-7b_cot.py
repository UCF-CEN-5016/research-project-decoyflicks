import pandas as pd

# Create a DataFrame with a mix of numeric and string (datetime) columns
data = {
    'numeric': [1, 2, 3],
    'date': ['01.01.2009 00:10:00', '01.01.2009 00:11:00', '01.01.2009 00:12:00']
}
df = pd.DataFrame(data)

# Attempt to compute correlation matrix (this will trigger the error)
try:
    df.corr()
except Exception as e:
    print("Error computing correlation matrix:", e)

# Convert datetime column to numeric (if applicable) or drop non-numeric columns
df_numeric = df.select_dtypes(include=['number'])
print("\nCorrelation matrix for numeric columns:")
print(df_numeric.corr())

# Convert datetime strings to timestamps (if needed)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date']).astype(int)  # Convert to Unix timestamp
    print("\nCorrelation matrix after converting 'date' column to timestamps:")
    print(df.corr())