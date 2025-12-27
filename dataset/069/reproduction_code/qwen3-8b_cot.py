import pandas as pd
import numpy as np

# Create a DataFrame with a mix of numeric and string (datetime) columns
df = pd.DataFrame({
    'numeric': [1, 2, 3],
    'date': ['01.01.2009 00:10:00', '01.01.2009 00:11:00', '01.01.2009 00:12:00']
})

# Attempt to compute correlation matrix (this will trigger the error)
df.corr()

# Convert datetime column to numeric (if applicable) or drop non-numeric columns
df = df.select_dtypes(include=[np.number])
df.corr()

# Convert datetime strings to timestamps (if needed)
df['date'] = pd.to_datetime(df['date']).astype(int)  # Convert to Unix timestamp
df.corr()