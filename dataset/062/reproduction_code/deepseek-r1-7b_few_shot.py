import pandas as pd

# Example showing DataFrame.append error

train_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['A', 'B', 'C']})
test_df = pd.DataFrame({'a': [4, 5, 6], 'b': ['D', 'E', 'F']})

# This will cause the AttributeError
try:
    all_data = train_df.append(test_df)
except AttributeError as e:
    print(f"Error: {e}")

# Corrected version using concat instead of append
all_data = pd.concat([train_df, test_df])