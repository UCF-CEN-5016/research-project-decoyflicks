import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load weather dataset (assuming this is your actual code)
df = pd.read_csv('weather_data.csv')

def show_heatmap(df):
    # This line causes the error
    plt.matshow(df.corr())

show_heatmap(df)