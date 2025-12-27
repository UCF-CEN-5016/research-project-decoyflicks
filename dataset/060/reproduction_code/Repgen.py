import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Constants
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
NUM_CLASSES = 3

# Create synthetic data
input_images = tf.random.normal([BATCH_SIZE] + list(IMG_SIZE) + [3], dtype=tf.float32)
target_masks = tf.random.uniform([BATCH_SIZE] + list(IMG_SIZE), maxval=NUM_CLASSES, dtype=tf.int32)

# Create base dataset
dataset = tf.data.Dataset.from_tensor_slices((input_images, target_masks))

# Define augmentation that will cause type mismatch
def problematic_augment(image, mask):
    # Convert image to int64 while mask remains int32
    augmented_image = tf.cast(image * 255, tf.int64)
    # Random flip that returns float32
    if tf.random.uniform([]) > 0.5:
        augmented_image = tf.cast(tf.image.flip_left_right(augmented_image), tf.float32)
    return augmented_image, mask

# Apply transformations that will trigger the bug
train_ds = (
    dataset
    .map(problematic_augment)  # This creates type mismatch
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# Simple model
model = keras.Sequential([
    layers.Input(shape=IMG_SIZE + (3,)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.Conv2D(NUM_CLASSES, 1)
])

# Compile and attempt to train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# This will raise a TypeError due to type mismatch in the data pipeline
try:
    model.fit(train_ds, epochs=1)
except TypeError as e:
    print("TypeError occurred as expected:")
    print(e)