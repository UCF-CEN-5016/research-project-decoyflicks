import pandas as pd
import matplotlib.pyplot as plt

# Bug Type: Data Type Mismatch
# Bug Description: DataFrame contains non-numeric columns causing ValueError when computing correlations

# Reproduction Code:

# DataFrame with mixed data types
df = pd.DataFrame({
    'date': ['01.01.2009 00:10:00', '01.01.2009 00:20:00'],
    'temp': [20, 22],
    'humidity': [60, 65]
})

# Attempt to compute correlation matrix
df_numeric = df.select_dtypes(include=['int64', 'float64'])  # Select only numeric columns for correlation
plt.matshow(df_numeric.corr())
plt.show()