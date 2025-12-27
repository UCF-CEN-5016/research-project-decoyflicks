import pandas as pd
import matplotlib.pyplot as plt

# Create a minimal DataFrame that reproduces the issue
data = {
    'date': ['01.01.2009 00:10:00', '01.01.2009 00:20:00', '01.01.2009 00:30:00'],
    'temperature': [12.5, 13.1, 12.8],
    'humidity': [45, 47, 43]
}
df = pd.DataFrame(data)

# Function that triggers the error (same as in original code)
def show_heatmap(data):
    plt.matshow(data.corr())
    plt.show()

# This will fail because 'date' column contains strings
show_heatmap(df)

def show_heatmap(data):
    # Exclude non-numeric columns
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    plt.matshow(numeric_data.corr())
    plt.show()