import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Sample data with datetime strings
data = np.array([[1, '01.01.2009 00:10:00'], [2, '02.02.2009 00:15:00'], [3, '03.03.2009 00:20:00']], dtype=object)
df = pd.DataFrame(data, columns=['value', 'datetime'])

# Convert datetime strings to actual datetime objects
df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M:%S')

# Calculate correlation matrix without including the datetime column
correlation_matrix = df[['value']].corr()

# Display the heatmap
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title('Correlation Heatmap')
plt.show()