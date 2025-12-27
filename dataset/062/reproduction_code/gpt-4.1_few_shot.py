import pandas as pd

# Create two simple DataFrames
train_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
test_df = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

# This used to work but now raises AttributeError in pandas 2.x:
# all_data = train_df.append(test_df)

# Uncommenting the line above causes:
# AttributeError: 'DataFrame' object has no attribute 'append'

# Correct approach:
all_data = pd.concat([train_df, test_df], ignore_index=True)
print(all_data)