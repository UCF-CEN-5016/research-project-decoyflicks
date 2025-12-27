import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set TensorFlow version
tf.__version__ = '2.10.0'

height, width, channels = 128, 128, 3
num_classes = 3
batch_size = 32

input_data = np.random.rand(batch_size, height, width, channels).astype(np.float32)
label_data = np.random.randint(0, num_classes, size=(batch_size, height, width, 1)).astype(np.int32)
label_data = np.squeeze(label_data, axis=-1)

model = keras.Sequential([
    layers.Input(shape=(height, width, channels)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(input_data, label_data, epochs=10, batch_size=batch_size)

print("Training accuracy:", history.history['accuracy'][-1])
print("Loss shape:", np.array(history.history['loss']).shape)