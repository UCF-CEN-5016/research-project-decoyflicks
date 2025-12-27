import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import pathlib
import random
import string
import re
import numpy as np

import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
import pandas as pd
import tensorflow_hub as hub
import tensorflow_decision_forests as tfdf
from keras import layers
from keras.layers import TextVectorization

# Simulated function to create a dataset
def create_dataset(dataframe):
    dataset = tf.data.Dataset.from_tensor_slices((dataframe["text"].to_numpy(), dataframe["target"].to_numpy()))
    dataset = dataset.batch(100)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Simulated function to load and prepare data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/IMvision12/Tweets-Classification-NLP/main/train.csv")
    df_shuffled = df.sample(frac=1, random_state=42)
    df_shuffled.drop(["id", "keyword", "location"], axis=1, inplace=True)
    df_shuffled.reset_index(inplace=True, drop=True)
    test_df = df_shuffled.sample(frac=0.1, random_state=42)
    train_df = df_shuffled.drop(test_df.index)
    return train_df, test_df

# Main code snippet to reproduce the bug
train_df, test_df = load_data()
train_ds = create_dataset(train_df)
test_ds = create_dataset(test_df)

sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")

inputs = layers.Input(shape=(), dtype=tf.string)
outputs = sentence_encoder_layer(inputs)
preprocessor = keras.Model(inputs=inputs, outputs=outputs)
model_1 = tfdf.keras.GradientBoostedTreesModel(preprocessing=preprocessor)

model_1.compile(metrics=["Accuracy", "Recall", "Precision", "AUC"])
model_1.fit(train_ds)

logs_1 = model_1.make_inspector().training_logs()
print(logs_1)

results = model_1.evaluate(test_ds, return_dict=True, verbose=0)
for name, value in results.items():
    print(f"{name}: {value:.4f}")