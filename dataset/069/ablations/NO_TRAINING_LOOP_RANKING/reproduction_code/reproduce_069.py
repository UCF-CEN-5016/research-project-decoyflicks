import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

df = pd.DataFrame({
    'date': ['01.01.2009 00:10:00', '01.01.2009 00:20:00'],
    'temperature': [20.5, 21.0],
    'humidity': [30, 35]
})

df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M:%S')
df['weather'] = ['sunny', 'cloudy']

def show_heatmap(data):
    plt.matshow(data.corr())
    plt.show()

show_heatmap(df)