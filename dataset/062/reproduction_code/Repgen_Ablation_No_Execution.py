import pandas as pd

# Download and read the CSV file into a DataFrame
df = pd.read_csv("https://raw.githubusercontent.com/IMvision12/Tweets-Classification-NLP/main/train.csv")

# Shuffle and drop unnecessary columns
df_shuffled = df.sample(frac=1, random_state=42)
df_shuffled.drop(["id", "keyword", "location"], axis=1, inplace=True)
df_shuffled.reset_index(inplace=True, drop=True)

# Split into train and test sets
test_df = df_shuffled.sample(frac=0.1, random_state=42)
train_df = df_shuffled.drop(test_df.index)

# Attempt to append train_df and test_df (will raise AttributeError)
all_data = train_df.append(test_df)