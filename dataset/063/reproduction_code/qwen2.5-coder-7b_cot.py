import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 4

def synthetic_data():
    # Create 20 synthetic images of shape (160,160,1) and integer labels
    images = tf.random.uniform((20, 160, 160, 1), dtype=tf.float32)
    labels = tf.range(20, dtype=tf.int64)
    return images, labels

images, labels = synthetic_data()
dataset = tf.data.Dataset.from_tensor_slices((images, labels))

def augment_fn(image, label):
    # Non-deterministically return either a float32 image or an int64-cast image.
    # This creates a situation where different elements in the pipeline may have
    # different dtypes for the image tensor (float32 vs int64).
    rnd = tf.random.uniform([], 0, 1)
    def as_float():
        return image, label
    def as_int():
        return tf.cast(image, tf.int64), label
    # Use tf.cond so the branch choice happens at graph/runtime; branches return different dtypes.
    return tf.cond(rnd < 0.5, as_float, as_int)

augmented_dataset = (
    dataset
    .shuffle(buffer_size=10)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# Iterate through the dataset to force pipeline execution (and trigger the error)
for batch in augmented_dataset.take(2):
    images_b, labels_b = batch
    print("Batch images dtype:", images_b.dtype, "labels dtype:", labels_b.dtype)