import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

df = pd.DataFrame({'date': ['01.01.2009 00:10:00', '01.01.2009 00:20:00'], 'value': ['10', '20']})
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M:%S')
df['numeric_value'] = [1.0, 2.0]

def show_heatmap(data):
    plt.matshow(data.corr())
    plt.show()

show_heatmap(df)