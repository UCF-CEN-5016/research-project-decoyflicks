import pandas as pd
import matplotlib.pyplot as plt

def plot_correlation_heatmap(dataframe):
    plt.matshow(dataframe.corr())
    plt.show()

def main():
    records = [
        ['01.01.2009 00:10:00', 'A', 5],
        ['02.01.2009 01:20:00', 'B', 4],
        ['03.01.2009 02:30:00', 'C', 3],
        ['04.01.2009 03:40:00', 'D', 2],
        ['05.01.2009 04:50:00', 'E', 1]
    ]
    dataframe = pd.DataFrame(records, columns=['datetime', 'category', 'value'])
    plot_correlation_heatmap(dataframe)

if __name__ == '__main__':
    main()