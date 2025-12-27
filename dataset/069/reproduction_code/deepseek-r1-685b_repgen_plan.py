import pandas as pd
import matplotlib.pyplot as plt

# Sample data with datetime strings
data = {
    'date': ['01.01.2009 00:10:00', '01.01.2009 00:20:00', '01.01.2009 00:30:00'],
    'temperature': [22.1, 22.3, 22.0],
    'humidity': [45, 46, 44]
}
df = pd.DataFrame(data)

# Function to show correlation heatmap
def show_heatmap(dataframe):
    data_for_corr = dataframe.drop('date', axis=1)  # Exclude datetime column for correlation
    plt.matshow(data_for_corr.corr())
    plt.show()

show_heatmap(df)  # Show correlation heatmap without datetime column.