import tensorflow as tf
from tensorflow import keras
import keras_cv

# Minimal model
model = keras.Sequential([
    keras.layers.Input(shape=(32, 32, 3)),
    keras.layers.Conv2D(16, 3, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
])

model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# Create a tf.data.Dataset from a generator (unknown cardinality)
def data_generator():
    while True:
        # Random image and label
        img = tf.random.uniform(shape=(32, 32, 3))
        label = tf.random.uniform(shape=(), minval=0, maxval=10, dtype=tf.int32)
        yield img, label

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
)

# Batch the dataset
dataset = dataset.batch(4)

# This will raise the error because dataset cardinality is unknown and steps_per_epoch is not set
model.fit(dataset, epochs=1)