import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a sample DataFrame with datetime values
data = {
    'Date': ['01.01.2009 00:10:00', '01.01.2009 00:20:00', '01.01.2009 00:30:00'],
    'Value': [10, 20, 30]
}
df = pd.DataFrame(data)

# Attempt to calculate correlation between columns
def show_heatmap(df):
    plt.matshow(df.corr())

# Call the function to reproduce the error
show_heatmap(df)