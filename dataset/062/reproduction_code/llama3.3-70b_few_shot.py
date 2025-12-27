import pandas as pd

# Sample DataFrames
train_df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

test_df = pd.DataFrame({
    'A': [7, 8, 9],
    'B': [10, 11, 12]
})

# Attempting to use the deprecated append method
try:
    all_data = train_df.append(test_df)
except AttributeError as e:
    print(f"Error: {e}")

# Correct way to concatenate DataFrames
all_data = pd.concat([train_df, test_df])
print(all_data)