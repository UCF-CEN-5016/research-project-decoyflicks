import pandas as pd
import matplotlib.pyplot as plt

# Sample data with datetime strings
data = {
    'date': ['01.01.2009 00:10:00', '01.01.2009 00:20:00', '01.01.2009 00:30:00'],
    'temperature': [22.1, 22.3, 22.0],
    'humidity': [45, 46, 44]
}
df = pd.DataFrame(data)

# Function that triggers the error
def show_heatmap(data):
    plt.matshow(data.corr())  # Fails when trying to correlate datetime strings
    plt.show()

# This will raise ValueError: could not convert string to float
show_heatmap(df)

# Correct approach would be to either:
# 1. Convert datetime strings to proper datetime objects first
# 2. Exclude datetime column from correlation calculation