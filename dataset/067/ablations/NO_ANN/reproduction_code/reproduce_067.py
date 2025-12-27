import os
import numpy as np
import tensorflow as tf
from keras import layers, models, utils

os.environ["KERAS_BACKEND"] = "tensorflow"
np.random.seed(42)

batch_size = 32
height, width, channels = 128, 128, 3
num_classes = 10

x_train = np.random.rand(batch_size, height, width, channels).astype("float32")
y_train = np.random.randint(0, num_classes, size=(batch_size, height, width))

model = models.Sequential([
    layers.Input(shape=(height, width, channels)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_split=0.1)

print("Final training loss:", history.history['loss'][-1])
print("Final training accuracy:", history.history['accuracy'][-1])
print(model.summary())