import tensorflow as tf

# Dummy dataset with int64 images
images = tf.constant([[[128, 255], [0, 128]], [[255, 0], [128, 0]]], dtype=tf.int64)
train_ds = tf.data.Dataset.from_tensor_slices(images)

# Augmentation function that converts to float32
def augment_fn(image):
    return tf.cast(image, tf.float32)

# Function that processes int64 input
def process_int64_inputs(image):
    return image

# Dataset pipeline with corrected types
augmented_train_ds = (
    train_ds.shuffle(2 * 16)  # BATCH_SIZE = 16
    .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(16)
    .map(process_int64_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Iterate over the dataset
for batch in augmented_train_ds:
    print(batch)