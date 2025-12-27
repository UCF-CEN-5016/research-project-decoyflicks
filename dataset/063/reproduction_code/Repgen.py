import tensorflow as tf
import tensorflow_datasets as tfds

# Constants
BATCH_SIZE = 32
IMG_HEIGHT = 160
IMG_WIDTH = 160

# Create synthetic dataset with mixed types
def create_mixed_type_dataset():
    # Create images as float32
    images = tf.random.uniform([BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3], 
                             dtype=tf.float32)
    # Create masks as int64
    masks = tf.random.uniform([BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 1], 
                            maxval=2,
                            dtype=tf.int64)
    return tf.data.Dataset.from_tensor_slices((images, masks))

# Problematic augmentation function that doesn't handle types consistently
def augment_fn(image, mask):
    # Image augmentation returns float32
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    
    # Mask operations maintain int64 type
    mask = tf.cast(mask, tf.int64)  # Explicit cast to trigger error
    return image, mask

# Create dataset with type mismatch
train_ds = create_mixed_type_dataset()

try:
    # This pipeline will trigger the type mismatch error
    augmented_ds = (train_ds
        .shuffle(BATCH_SIZE * 2)
        .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE))
    
    # Try to iterate through the dataset
    next(iter(augmented_ds))
    
except tf.errors.InvalidArgumentError as e:
    print("Expected error occurred due to type mismatch:")
    print(e)