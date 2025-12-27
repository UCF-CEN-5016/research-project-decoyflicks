import pandas as pd
import numpy as np

# Create DataFrames
train_df = pd.DataFrame({
    'A': np.random.randint(0, 100, size=10),
    'B': np.random.randint(0, 100, size=10)
})

test_df = pd.DataFrame({
    'A': np.random.randint(0, 100, size=5),
    'B': np.random.randint(0, 100, size=5)
})

# Function to concatenate DataFrames using deprecated append method
def concatenate_dataframes(train_df, test_df):
    return train_df.append(test_df)

# Call the function
all_data = concatenate_dataframes(train_df, test_df)