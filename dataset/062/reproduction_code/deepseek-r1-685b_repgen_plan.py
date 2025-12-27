import pandas as pd

# Sample dataframes (minimal example)
train_df = pd.DataFrame({'text': ['train sample 1', 'train sample 2']})
test_df = pd.DataFrame({'text': ['test sample 1', 'test sample 2']})

# Concatenate dataframes using pd.concat
combined_df = pd.concat([train_df, test_df], ignore_index=True)

print("Successfully combined using concat():")
print(combined_df)