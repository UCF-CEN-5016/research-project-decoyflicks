import pandas as pd

# Create two sample DataFrames
train_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
test_df = pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]})

# Attempt to append DataFrames
all_data = train_df.append(test_df)

import pandas as pd

# Create two sample DataFrames
train_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
test_df = pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]})

# Use pandas.concat to combine DataFrames
all_data = pd.concat([train_df, test_df])