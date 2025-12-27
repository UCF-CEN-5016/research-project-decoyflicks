import pandas as pd
import matplotlib.pyplot as plt

def show_heatmap(data):
    plt.matshow(data.corr())
    plt.show()

data = {
    'date': ['01.01.2009 00:10:00', '01.01.2009 00:20:00', '01.01.2009 00:30:00'],
    'value': [10.5, 11.2, 12.1]
}
df = pd.DataFrame(data)
show_heatmap(df)