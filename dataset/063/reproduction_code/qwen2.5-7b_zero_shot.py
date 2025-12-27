import tensorflow as tf
import numpy as np

BATCH_SIZE = 2
AUTOTUNE = tf.data.AUTOTUNE

# Create dummy dataset
def generate_random_image():
    return np.random.randint(0, 256, (160, 160, 3), np.int64)

def generate_random_label():
    return np.random.randint(0, 2, (160, 160, 1), np.int64)

images = tf.data.Dataset.from_tensor_slices([generate_random_image() for _ in range(10)])
labels = tf.data.Dataset.from_tensor_slices([generate_random_label() for _ in range(10)])
train_ds = tf.data.Dataset.zip((images, labels))

def augment_fn(image, label):
    # Convert image to float32
    image = tf.cast(image, tf.float32)
    return image, label

def preprocess_fn(image, label):
    # Suppose this function is supposed to process the image as int64, but it's in float32
    # This would cause an error if the model expects the image to be in int64, but it's in float32
    return tf.cast(image, tf.int64), label

augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(preprocess_fn)
    .prefetch(buffer_size=AUTOTUNE)
)

# Example usage
for batch in augmented_train_ds.take(1):
    print(batch)