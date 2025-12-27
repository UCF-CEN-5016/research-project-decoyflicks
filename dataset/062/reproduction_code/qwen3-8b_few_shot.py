import pandas as pd

# Create sample DataFrames
train_df = pd.DataFrame({'a': [1, 2]})
test_df = pd.DataFrame({'a': [3, 4]})

# Attempting to use deprecated DataFrame.append() method
all_data = train_df.append(test_df)

print("Combined DataFrame:")
print(all_data)