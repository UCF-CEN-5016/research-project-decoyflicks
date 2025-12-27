import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)

root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")

df = pd.DataFrame(x_train[:100])
df.columns = [f"feature_{i}" for i in range(df.shape[1])]
df['timestamp'] = ['01.01.2009 00:10:00' for _ in range(100)]

# Convert non-numeric values to NaN
df = df.apply(pd.to_numeric, errors='coerce')

plt.matshow(df.corr(), fignum=1)
plt.colorbar()
plt.show()