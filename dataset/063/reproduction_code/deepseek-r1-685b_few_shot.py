import tensorflow as tf

# Sample augmentation function that creates type mismatch
def augment_fn(image, mask):
    # Randomly applies different augmentations with different output types
    image = tf.image.random_brightness(image, max_delta=0.2)  # Keeps float32
    mask = tf.image.random_flip_left_right(mask)  # Keeps int64
    return image, mask

# Sample unpacking function
def unpackage_inputs(image, mask):
    return image, mask

# Create dummy dataset
BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE
dummy_images = tf.random.uniform((32, 160, 160, 1), dtype=tf.float32)
dummy_masks = tf.random.uniform((32, 160, 160, 1), minval=0, maxval=10, dtype=tf.int64)

train_ds = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_masks))

# Pipeline that triggers the error
augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=AUTOTUNE)
)

# Try to iterate (will trigger the error)
for batch in augmented_train_ds:
    images, masks = batch
    print(images.shape, masks.shape)