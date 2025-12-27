import pandas as pd

# Minimal example to reproduce the bug fix
train_df = pd.DataFrame({'A': [1, 2], 'B': ['a', 'b']})
test_df = pd.DataFrame({'A': [3, 4], 'B': ['c', 'd']})

all_data = pd.concat([train_df, test_df])