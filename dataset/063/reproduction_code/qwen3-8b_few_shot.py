import tensorflow as tf

# Dummy dataset with int64 images
images = tf.constant([[[128, 255], [0, 128]], [[255, 0], [128, 0]]], dtype=tf.int64)
train_ds = tf.data.Dataset.from_tensor_slices(images)

# Augmentation function that converts to float32
def augment_fn(image):
    return tf.cast(image, tf.float32)

# Function expecting int64 input
def unpackage_inputs(image):
    return tf.cast(image, tf.int64)  # This will cause type mismatch

# Dataset pipeline that triggers the error
augmented_train_ds = (
    train_ds.shuffle(2 * 16)  # BATCH_SIZE = 16
    .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(16)
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Attempt to get an element (this will raise the type mismatch error)
for batch in augmented_train_ds:
    print(batch)