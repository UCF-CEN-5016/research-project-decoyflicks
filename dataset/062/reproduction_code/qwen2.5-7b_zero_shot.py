import pandas as pd

train_data = {'a': [1, 2]}
test_data = {'a': [3, 4]}

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

all_data = pd.concat([train_df, test_df], ignore_index=True)