import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = [
    ['01.01.2009 00:10:00', 'A', 5],
    ['02.01.2009 01:20:00', 'B', 4],
    ['03.01.2009 02:30:00', 'C', 3],
    ['04.01.2009 03:40:00', 'D', 2],
    ['05.01.2009 04:50:00', 'E', 1]
]

df = pd.DataFrame(data, columns=['datetime', 'category', 'value'])
show_heatmap(df)

def show_heatmap(df):
    plt.matshow(df.corr())
    plt.show()