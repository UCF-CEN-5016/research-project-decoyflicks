import pandas as pd

train_df = pd.DataFrame({'a': [1, 2]})
test_df = pd.DataFrame({'a': [3, 4]})
all_data = train_df.append(test_df)