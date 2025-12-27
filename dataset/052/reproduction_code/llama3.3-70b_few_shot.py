import tensorflow as tf
from tensorflow import keras
import numpy as np

# Define a sample model
ner_model = keras.Sequential([
    keras.layers.Embedding(input_dim=1000, output_dim=128),
    keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
    keras.layers.TimeDistributed(keras.layers.Dense(8, activation='softmax'))
])

# Compile the model
ner_model.compile(optimizer="adam", loss='sparse_categorical_crossentropy')

# Define a sample dataset
train_dataset = tf.data.Dataset.from_tensor_slices(
    (np.random.randint(0, 1000, size=(32, 10)), np.random.randint(0, 8, size=(32, 10)))
).batch(32)

# Train the model
ner_model.fit(train_dataset, epochs=10)