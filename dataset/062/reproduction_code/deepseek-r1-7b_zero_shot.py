import pandas as pd
import numpy as np

# Create two sample DataFrames
train_df = pd.DataFrame({'A': np.random.randn(5), 'B': np.random.randn(5)})
test_df = pd.DataFrame({'A': np.random.randn(5), 'B': np.random.randn(5)})

try:
    all_data = train_df.append(test_df)
except AttributeError as e:
    print(f"AttributeError: {e}")

# The correct way using concat
all_data = pd.concat([train_df, test_df], ignore_index=True)