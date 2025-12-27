import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
import tensorflow_datasets as tfds  # Add this line to fix the undefined variable error

# Constants
HEIGHT = 180
WIDTH = 180
NUM_CLASSES = 37
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
AUTOTUNE = tf.data.AUTOTUNE

# Load dataset
train_ds, val_ds = tfds.load(
    name="oxford_pets",
    as_supervised=True,
    split=["train[:128]", "validation[:32]"],
)

# Preprocess data
rescale_images_and_correct_masks = lambda d: (
    (tf.image.resize(d["image"], [HEIGHT, WIDTH]) / 255.0),
    tf.cast(d["segmentation_mask"] - 1, tf.int64)  # Corrected dtype here
)

train_ds = train_ds.map(rescale_images_and_correct_masks, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(rescale_images_and_correct_masks, num_parallel_calls=AUTOTUNE)

# Unpackage inputs
def unpackage_inputs(ds):
    return ds["image"], ds["segmentation_mask"]

train_ds = train_ds.map(unpackage_inputs)
val_ds = val_ds.map(unpackage_inputs)

# Augmentation function
def augment_fn(images, masks):
    images = tf.image.resize(images, [HEIGHT, WIDTH])
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    images = tf.image.rot90(images, k=random.randint(0, 3))
    images = tf.ensure_shape(images, (None, None, 3))
    return images, masks

augmented_train_ds = train_ds.shuffle(BATCH_SIZE * 2).map(
    augment_fn, num_parallel_calls=AUTOTUNE
)
augmented_train_ds = augmented_train_ds.batch(BATCH_SIZE)

# Callback class
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            images, masks = next(iter(augmented_train_ds))
            plt.figure(figsize=(10, 10))
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                plt.imshow(images[i])
                plt.title(f"Epoch {epoch + 1}")
            plt.show()
            plt.close()

# Define model
model = layers.Sequential([
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.SeparableConv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.SeparableConv2D(256, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(NUM_CLASSES, (1, 1), activation="softmax")
])

# Create callback instance
display_callback = DisplayCallback()

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(augmented_train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[display_callback])