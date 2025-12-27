import pandas as pd
import matplotlib.pyplot as plt

# Sample data with datetime strings
data = {
    'date': ['01.01.2009 00:10:00', '01.01.2009 00:20:00', '01.01.2009 00:30:00'],
    'temperature': [22.1, 22.3, 22.0],
    'humidity': [45, 46, 44]
}
df = pd.DataFrame(data)

# Function that triggers the error
def show_heatmap(data):
    # Exclude datetime column from correlation calculation
    corr_data = data.drop('date', axis=1)
    plt.matshow(corr_data.corr())
    plt.show()

# This will now work without errors
show_heatmap(df)