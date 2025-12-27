import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

inputs = keras.Input(shape=(4,), dtype=tf.int32)
x = layers.Embedding(input_dim=10, output_dim=4)(inputs)
x = layers.Softmax()(x)
outputs = layers.Dense(3)(x)
model = keras.Model(inputs, outputs)

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss)

x_train = np.random.randint(0, 10, (2, 4))
y_train = np.random.randint(0, 3, (2, 4))

model.fit(x_train, y_train, epochs=1)