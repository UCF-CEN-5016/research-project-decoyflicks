import pandas as pd
import matplotlib.pyplot as plt

data = {
    'date': ['01.01.2009 00:10:00', '01.01.2009 00:20:00', '01.01.2009 00:30:00'],
    'value1': [1.0, 2.0, 3.0],
    'value2': [4.0, 5.0, 6.0]
}
df = pd.DataFrame(data)

def show_heatmap(data):
    plt.matshow(data.corr())
    plt.show()

show_heatmap(df)