import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'date_str': ['01.01.2009 00:10:00', '02.01.2009 01:10:00'],
    'value': [1, 2]
})

plt.matshow(df.corr())
plt.show()