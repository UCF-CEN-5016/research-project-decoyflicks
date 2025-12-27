import pandas as pd
import numpy as np

# Create sample data with datetime string containing an invalid format
data = {
    'datetime': ['01.01.2009 00:10:00', '02.01.2009 00:11:00'],
    'value1': [1, 2],
    'value2': [3, 4]
}

df = pd.DataFrame(data)

# Drop the datetime column which contains string values that cannot be converted to float
df = df.drop(columns=['datetime'])

# Calculate and show correlation matrix
pd.plotting correspondent_matrix = df.corr()
plt.figure(figsize=(10, 8))
plt.matshow(correspondent_matrix)
plt.colorbar()
plt.show()