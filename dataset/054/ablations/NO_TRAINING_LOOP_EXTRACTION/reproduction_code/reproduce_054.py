import keras
import numpy as np
import tensorflow as tf

np.random.seed(42)

batch_size = 32
input_shape = (64, 64, 3)

train_images = np.random.rand(1000, *input_shape)
train_labels = np.random.randint(0, 10, size=(1000,))
test_images = np.random.rand(200, *input_shape)
test_labels = np.random.randint(0, 10, size=(200,))

model = keras.Sequential([
    keras.layers.Input(shape=input_shape),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))