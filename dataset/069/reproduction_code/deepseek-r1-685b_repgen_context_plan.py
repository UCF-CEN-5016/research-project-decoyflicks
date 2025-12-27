import pandas as pd
import matplotlib.pyplot as plt

# Sample data with datetime strings
data = {
    'date': ['01.01.2009 00:10:00', '01.01.2009 00:20:00', '01.01.2009 00:30:00'],
    'temperature': [22.1, 22.3, 22.0],
    'humidity': [45, 46, 44]
}
df = pd.DataFrame(data)

# Function to remove datetime column
def remove_datetime_column(data):
    return data.drop('date', axis=1)

# Function to show heatmap of correlation matrix
def show_heatmap(data):
    plt.matshow(remove_datetime_column(data).corr())
    plt.show()

show_heatmap(df)