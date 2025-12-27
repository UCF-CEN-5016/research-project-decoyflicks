import pandas as pd

# Sample dataframes (minimal example)
train_df = pd.DataFrame({'text': ['train sample 1', 'train sample 2']})
test_df = pd.DataFrame({'text': ['test sample 1', 'test sample 2']})

# Check for pandas version and combine dataframes
try:
    combined_df = train_df.append(test_df)
    print("Successfully combined using append:")
except AttributeError as e:
    print(f"Error occurred while using append: {e}")

# Alternative method using concat
combined_df = pd.concat([train_df, test_df])
print("Successfully combined using concat:")
print(combined_df)