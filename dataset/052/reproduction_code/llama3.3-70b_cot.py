import tensorflow as tf
from tensorflow import keras
import numpy as np

# Set up the minimal environment
tf.config.set_soft_device_placement(True)

# Define the model
def create_model():
    model = keras.Sequential([
        keras.layers.Embedding(input_dim=1000, output_dim=128, input_length=100),
        keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
        keras.layers.TimeDistributed(keras.layers.Dense(64, activation='relu')),
        keras.layers.TimeDistributed(keras.layers.Dense(8, activation='softmax'))
    ])
    return model

# Create the model
ner_model = create_model()

# Compile the model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
ner_model.compile(optimizer="adam", loss=loss)

# Define the dataset
def generate_dataset():
    # Generate a sample dataset
    inputs = np.random.randint(0, 1000, size=(32, 100))
    labels = np.random.randint(0, 8, size=(32, 100))
    return inputs, labels

# Generate the dataset
train_dataset = generate_dataset()

# Trigger the bug
ner_model.fit(train_dataset[0], train_dataset[1], epochs=10)