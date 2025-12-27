import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

df = pd.DataFrame({'date': pd.date_range(start='2009-01-01', periods=100, freq='10T'), 'value': np.random.rand(100)})
df['date'] = df['date'].dt.strftime('%d.%m.%Y %H:%M:%S')
df.set_index('date', inplace=True)
df['mixed'] = ['text'] * 100

def show_heatmap(data):
    plt.matshow(data.corr())

show_heatmap(df)