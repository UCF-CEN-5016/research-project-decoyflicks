import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras, layers

# Set seed for reproducibility
np.random.seed(42)

# Download and read data
url = "https://raw.githubusercontent.com/IMvision12/Tweets-Classification-NLP/main/train.csv"
df = pd.read_csv(url)

# Shuffle and preprocess data
df_shuffled = df.sample(frac=1, random_state=42)
df_shuffled.drop(["id", "keyword", "location"], axis=1, inplace=True)
df_shuffled.reset_index(drop=True, inplace=True)

# Split into train and test sets
test_df = df_shuffled.sample(frac=0.1, random_state=42)
train_df = df_shuffled.drop(test_df.index)

# Create dataset function
def create_dataset(dataframe):
    return tf.data.Dataset.from_tensor_slices((dataframe["text"].to_numpy(), dataframe["target"].to_numpy())).batch(100).prefetch(tf.data.AUTOTUNE)

# Create datasets
train_ds = create_dataset(train_df)
test_ds = create_dataset(test_df)

# Import TensorFlow Decision Forests
import tensorflow_decision_forests as tfdf

# Define sentence encoder layer
sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")

# Create model_1
inputs = layers.Input(shape=(), dtype=tf.string)
outputs = sentence_encoder_layer(inputs)
preprocessor = keras.Model(inputs=inputs, outputs=outputs)
model_1 = tfdf.keras.GradientBoostedTreesModel(preprocessing=preprocessor)

# Compile and train model_1
model_1.compile(metrics=["Accuracy", "Recall", "Precision", "AUC"])
model_1.fit(train_ds)

# Concatenate train_df and test_df using pandas.concat instead of deprecated append method
all_data = pd.concat([train_df, test_df], ignore_index=True)

# Assert AttributeError when trying to use the deprecated append() method
try:
    all_data = train_df.append(test_df)
except AttributeError as e:
    print(e)