import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE

# Create dummy datasets
images = tf.data.Dataset.from_tensor_slices(
    [np.random.randint(0, 256, (160, 160, 3), np.uint8) for _ in range(100)]
)
masks = tf.data.Dataset.from_tensor_slices(
    [np.random.randint(0, 2, (160, 160, 1), np.int64) for _ in range(100)]
)

dataset = tf.data.Dataset.zip((images, masks))

def augment_fn(image, mask):
    # Convert image to float32 (common for neural network input)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    return image, mask  # Mask remains int64

def unpackage_inputs(x):
    # Ensure types are consistent before feeding into model
    image, mask = x
    return image, mask  # No further type conversion needed

# Apply the pipeline
augmented_train_ds = (
    dataset
    .shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=AUTOTUNE)
)

# Example: Feed data into a model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(160, 160, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2, activation='softmax')  # Output matches mask shape
])

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(augmented_train_ds, epochs=5)