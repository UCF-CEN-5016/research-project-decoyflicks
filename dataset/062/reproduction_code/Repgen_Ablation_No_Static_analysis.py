import pandas as pd

# Download and read the CSV file
df = pd.read_csv("https://raw.githubusercontent.com/IMvision12/Tweets-Classification-NLP/main/train.csv")

# Shuffle the DataFrame
df_shuffled = df.sample(frac=1, random_state=42)

# Drop unnecessary columns
df_shuffled.drop(["id", "keyword", "location"], axis=1, inplace=True)
df_shuffled.reset_index(drop=True, inplace=True)

# Split into training and test sets
test_df = df_shuffled.sample(frac=0.1, random_state=42)
train_df = df_shuffled.drop(test_df.index)

# Define the create_dataset function
def create_dataset(dataframe):
    dataset = tf.data.Dataset.from_tensor_slices((dataframe["text"].to_numpy(), dataframe["target"].to_numpy()))
    dataset = dataset.batch(100).prefetch(tf.data.AUTOTUNE)
    return dataset

# Create datasets
train_ds = create_dataset(train_df)
test_ds = create_dataset(test_df)

# Append DataFrames (this will trigger the bug)
all_data = train_df.append(test_df)