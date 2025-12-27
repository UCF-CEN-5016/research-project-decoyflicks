import pandas as pd

train_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
test_df = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
all_data = train_df.append(test_df)