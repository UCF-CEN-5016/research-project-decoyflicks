import pandas as pd

# Create two minimal DataFrames
train_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
test_df = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

# This line triggers the AttributeError in pandas 2.0.3+
all_data = train_df.append(test_df)  # Deprecated, removed in pandas 2.x

print(all_data)