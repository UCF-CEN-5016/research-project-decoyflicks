import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Minimal setup: Create a DataFrame with a datetime column and a numerical column
df = pd.DataFrame({
    'Date': ['01.01.2009 00:10:00', '01.01.2009 01:10:00', '01.01.2009 02:10:00'],
    'Value': [10, 20, 30]
})

# Attempt to calculate and plot the correlation, which triggers the bug
def show_heatmap(data):
    # This line triggers the bug because it tries to convert all columns to float
    plt.matshow(data.corr())

# Call the function with our DataFrame
show_heatmap(df)