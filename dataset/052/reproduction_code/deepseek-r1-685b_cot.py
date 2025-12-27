import tensorflow as tf
import keras
from keras import layers
import numpy as np

# Verify we're using Keras 3
print("Keras version:", keras.__version__)

# Sample data - minimal NER setup
vocab_size = 100
num_tags = 5
seq_length = 10
batch_size = 2

x_train = np.random.randint(0, vocab_size, size=(100, seq_length))
y_train = np.random.randint(0, num_tags, size=(100, seq_length))

# Simple transformer-based NER model
def build_model():
    inputs = keras.Input(shape=(seq_length,), dtype=tf.int32)
    x = layers.Embedding(vocab_size, 64)(inputs)
    x = layers.TransformerEncoder(
        num_heads=2,
        intermediate_dim=64,
        dropout=0.1
    )(x)
    outputs = layers.Dense(num_tags, activation="softmax")(x)
    return keras.Model(inputs, outputs)

model = build_model()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# This will trigger the error
try:
    model.fit(x_train, y_train, batch_size=batch_size, epochs=1)
except Exception as e:
    print("Error occurred:")
    print(e)