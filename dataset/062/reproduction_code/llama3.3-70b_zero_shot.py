import pandas as pd

train_df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

test_df = pd.DataFrame({
    'A': [7, 8, 9],
    'B': [10, 11, 12]
})

all_data = train_df.append(test_df)