import pandas as pd

# Create two sample DataFrames
data1 = {'Name': ['John', 'Anna', 'Peter', 'Linda'],
        'Age': [28, 24, 35, 32]}
train_df = pd.DataFrame(data1)

data2 = {'Name': ['Tom', 'Jessica', 'Michael', 'Sophia'],
        'Age': [22, 29, 36, 31]}
test_df = pd.DataFrame(data2)

# Attempt to use the append method (this will raise the AttributeError)
try:
    all_data = train_df.append(test_df)
except AttributeError as e:
    print(f"Error: {e}")

# Correct way to concatenate DataFrames using concat
all_data = pd.concat([train_df, test_df])
print(all_data)