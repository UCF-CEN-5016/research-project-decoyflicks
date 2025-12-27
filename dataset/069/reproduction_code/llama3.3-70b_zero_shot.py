import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show_heatmap(df):
    plt.matshow(df.corr())
    plt.show()

data = {
    'Date': ['01.01.2009 00:10:00', '01.01.2009 00:20:00', '01.01.2009 00:30:00'],
    'Value': [10, 20, 30]
}

df = pd.DataFrame(data)
show_heatmap(df)