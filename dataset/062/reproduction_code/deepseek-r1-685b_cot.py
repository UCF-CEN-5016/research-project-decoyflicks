import pandas as pd

# Setup environment - pandas >= 2.0
print(f"pandas version: {pd.__version__}")  # Should be >= 2.0

# Create sample DataFrames
train_df = pd.DataFrame({'text': ['train sample 1', 'train sample 2']})
test_df = pd.DataFrame({'text': ['test sample 1', 'test sample 2']})

# Trigger the bug (will fail on pandas >= 2.0)
try:
    all_data = train_df.append(test_df)  # This will raise AttributeError
except AttributeError as e:
    print(f"Error occurred: {e}")

# Correct alternative using concat
all_data = pd.concat([train_df, test_df])
print("Successfully concatenated using pd.concat:")
print(all_data)