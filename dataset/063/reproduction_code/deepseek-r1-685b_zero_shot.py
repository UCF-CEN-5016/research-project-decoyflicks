import tensorflow as tf
import numpy as np

BATCH_SIZE = 2
AUTOTUNE = tf.data.AUTOTUNE

def augment_fn(image, mask):
    image = tf.image.random_flip_left_right(image)
    mask = tf.image.random_flip_left_right(mask)
    return image, mask

def unpackage_inputs(image, mask):
    return image, mask

def create_dummy_dataset():
    images = np.random.rand(10, 160, 160, 3).astype(np.float32)
    masks = np.random.randint(0, 2, size=(10, 160, 160, 1)).astype(np.int64)
    return tf.data.Dataset.from_tensor_slices((images, masks))

train_ds = create_dummy_dataset()

augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

for batch in augmented_train_ds:
    pass