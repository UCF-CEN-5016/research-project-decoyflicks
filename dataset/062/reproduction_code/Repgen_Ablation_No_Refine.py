import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf

url = "https://raw.githubusercontent.com/IMvision12/Tweets-Classification-NLP/main/train.csv"
df = pd.read_csv(url)
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

test_df = df_shuffled.sample(frac=0.1, random_state=42)
train_df = df_shuffled.drop(test_df.index)

def create_dataset(dataframe):
    dataset = tf.data.Dataset.from_tensor_slices((dataframe["text"].to_numpy(), dataframe["target"].to_numpy()))
    dataset = dataset.batch(100).prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = create_dataset(train_df)
test_ds = create_dataset(test_df)

model_1 = tfdf.keras.GradientBoostedTreesModel()
model_1.compile(metrics=["Accuracy", "Recall", "Precision", "AUC"])
model_1.fit(train_ds, epochs=5)  # Added epochs parameter for the fit method

model_2 = tfdf.keras.GradientBoostedTreesModel()
model_2.compile(metrics=["Accuracy", "Recall", "Precision", "AUC"])
model_2.fit(train_ds, epochs=5)  # Added epochs parameter for the fit method