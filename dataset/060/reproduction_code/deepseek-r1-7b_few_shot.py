import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_next_image

# Cast masks to float32 for consistency with images
def preprocess_image(image, mask):
    return (tf.cast(image, dtype=tf.float32), 
            tf.cast(mask, dtype=tf.float32))

train_ds = ...  # Your existing training dataset

# Modified preprocessing pipeline
augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
)

resized_val_ds = (
    val_ds.map(resize_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(lambda x: (array_to_next_image(x[0]), array_to_next_image(x[1])), num_parallel_calls=AUTOTUNE)  # Cast masks to float32
)

# Ensure masks are properly casted during augmentation
augmented_train_ds = train_ds.map(preprocess_image).shuffle(BATCH_SIZE * 2)
augmented_train_ds = augmented_train_ds.map(augment_fn, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)