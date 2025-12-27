import pandas as pd
import matplotlib.pyplot as plt

# Sample dataframe with a datetime column as string and numeric columns
data = {
    'datetime': ['01.01.2009 00:10:00', '01.01.2009 00:20:00', '01.01.2009 00:30:00'],
    'temperature': [22.5, 23.0, 22.8],
    'humidity': [30, 35, 32]
}

df = pd.DataFrame(data)

# Attempt to plot correlation matrix, fails due to datetime string column
plt.matshow(df.corr())  # This line raises ValueError
plt.show()