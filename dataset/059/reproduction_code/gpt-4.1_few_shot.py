import tensorflow as tf
import keras_cv

# Minimal dataset with dict targets
def dummy_dataset():
    images = tf.random.uniform((4, 64, 64, 3))
    # targets as dict, e.g. for object detection
    targets = {
        "boxes": tf.random.uniform((4, 10, 4)),
        "classes": tf.random.uniform((4, 10), maxval=20, dtype=tf.int32),
    }
    return tf.data.Dataset.from_tensor_slices((images, targets)).batch(2)

# Simple model expecting only images and a single output tensor
inputs = tf.keras.Input(shape=(64, 64, 3))
x = tf.keras.layers.Conv2D(8, 3)(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="mse")

dataset = dummy_dataset()

# This will raise an error because targets are dicts but model expects arrays
model.fit(dataset, epochs=1)