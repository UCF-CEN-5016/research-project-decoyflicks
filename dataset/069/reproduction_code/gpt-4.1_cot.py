import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create minimal DataFrame with datetime strings and numeric data
data = {
    "datetime": ["01.01.2009 00:10:00", "01.01.2009 00:20:00", "01.01.2009 00:30:00"],
    "temperature": [20.5, 21.0, 19.8],
    "humidity": [30, 35, 33]
}

df = pd.DataFrame(data)

# Attempt to plot correlation matrix on whole DataFrame (including datetime string column)
plt.matshow(df.corr())
plt.title("Correlation matrix")
plt.show()