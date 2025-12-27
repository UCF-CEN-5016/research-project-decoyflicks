import tensorflow as tf

# Sample augmentation function that creates type mismatch
def augment_fn(image, mask):
    # Randomly applies different augmentations with different output types
    image = tf.image.random_brightness(image, max_delta=0.2)  # Keeps float32
    mask = tf.image.random_flip_left_right(tf.cast(mask, tf.float32))  # Convert mask to float32 for consistency
    return image, mask

# Create dummy dataset
BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE
dummy_images = tf.random.uniform((32, 160, 160, 1), dtype=tf.float32)
dummy_masks = tf.random.uniform((32, 160, 160, 1), minval=0, maxval=10, dtype=tf.int64)

train_ds = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_masks))

# Pipeline with improved structure
def process_data(image, mask):
    image, mask = augment_fn(image, mask)
    return image, mask

def configure_for_performance(ds, batch_size):
    ds = ds.shuffle(batch_size * 2)
    ds = ds.map(process_data, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

augmented_train_ds = configure_for_performance(train_ds, BATCH_SIZE)

# Iterate through the dataset
for batch in augmented_train_ds:
    images, masks = batch
    print(images.shape, masks.shape)