import pandas as pd
import numpy as np

# Set up minimal environment
df = pd.DataFrame({
    'numeric_column': [1.0, 2.0, 3.0],
    'string_column': ['01.01.2009 00:10:00', '02.02.2010 12:30:00', '03.03.2015 14:45:00'],
    'another_numeric_column': [4.0, 5.0, 6.0]
})

# Add triggering conditions
df['string_column'] = df['string_column'].astype(str)  # Ensure the column is of string type

try:
    plt.matshow(df.corr())
except ValueError as e:
    print(f"Error: {e}")