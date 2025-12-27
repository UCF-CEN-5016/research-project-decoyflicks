import pandas as pd

# Sample dataframes
train_df = pd.DataFrame({'text': ['example1', 'example2'], 'target': [0, 1]})
test_df = pd.DataFrame({'text': ['example3', 'example4'], 'target': [1, 0]})

# Concatenating DataFrames instead of using deprecated append method
all_data = pd.concat([train_df, test_df], ignore_index=True)

print(all_data)