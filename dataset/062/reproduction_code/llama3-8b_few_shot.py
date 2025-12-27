import pandas as pd

# Create two DataFrames
train_df = pd.DataFrame({'A': [1, 2, 3]})
test_df = pd.DataFrame({'B': [4, 5, 6]})

# Attempt to concatenate using deprecated append method
all_data = train_df.append(test_df)

print("Concatenated data:")
print(all_data)