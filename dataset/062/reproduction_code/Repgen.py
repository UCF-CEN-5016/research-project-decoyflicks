import pandas as pd

# Create sample DataFrames
train_df = pd.DataFrame({
    'text': ['example1', 'example2'],
    'target': [0, 1]
})

test_df = pd.DataFrame({
    'text': ['example3', 'example4'],
    'target': [1, 0]
})

# This will raise the AttributeError due to deprecated append method
try:
    print("Attempting to use deprecated append method...")
    all_data = train_df.append(test_df)
except AttributeError as e:
    print("\nAttributeError caught as expected:")
    print(e)
    print("\nThis error occurs because .append() was removed in pandas 2.0.0+")

# The correct way would be:
print("\nCorrect way using pd.concat:")
all_data = pd.concat([train_df, test_df], ignore_index=True)
print(all_data)