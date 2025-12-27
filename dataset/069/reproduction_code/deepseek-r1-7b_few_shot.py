import pandas as pd
import numpy as np

# Create sample DataFrame with datetime strings
data = {
    'date': ['01.01.2009 00:10:00', '02.01.2009 00:10:00', 
             '03.01.2009 00:10:00', '04.01.2009 00:10:00'],
    'temperature': [5, 7, 8, 6]
}

df = pd.DataFrame(data)

try:
    # Attempt to compute the correlation between numeric and datetime columns
    print("Correlation matrix:")
    print(df.corr())
except Exception as e:
    print(f"An error occurred: {e}")