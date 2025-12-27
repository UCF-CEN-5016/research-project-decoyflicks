import pandas as pd

# Create sample DataFrames
train_df = pd.DataFrame({'a': [1, 2]})
test_df = pd.DataFrame({'a': [3, 4]})

# Concatenate DataFrames along rows
all_data = pd.concat([train_df, test_df])

print("Combined DataFrame:")
print(all_data)