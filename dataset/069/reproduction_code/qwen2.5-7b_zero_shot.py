import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame
data = {'date_str': ['01.01.2009 00:10:00', '02.01.2009 01:10:00'],
        'value': [1, 2]}
df = pd.DataFrame(data)

# Display the correlation matrix using matshow
plt.matshow(df.corr())
plt.show()