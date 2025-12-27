import tensorflow as tf

# Define a simple data pipeline
BATCH_SIZE = 32

# Create sample dataset
train_ds = tf.data.Dataset.from_tensor_slices(
    (tf.random.normal([100, 160, 160, 3]), tf.random.normal([100, 160, 160, 1]))
).batch(BATCH_SIZE)

# Define augment function with inconsistent data types
def augment_fn(image, mask):
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.int64)  # Inconsistent data type
    return image, mask

# Define unpackage function
def unpackage_inputs(x, y):
    return x, y

# Apply augmentation and cause error
augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(lambda x, y: (tf.image.random_flip_left_right(x), y))  # Apply random flip
    .map(augment_fn)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Iterate over dataset to trigger error
for batch in augmented_train_ds:
    image, mask = batch
    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
    break