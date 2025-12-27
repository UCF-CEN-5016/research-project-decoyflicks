import tensorflow as tf
from keras import layers

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def augment_fn(image):
    image = tf.image.random_flip_left_right(image)
    return image

def unpackage_inputs(inputs):
    return inputs

train_ds = tf.data.Dataset.from_tensor_slices(tf.random.uniform((1000, 160, 160, 1), dtype=tf.float32))

augmented_train_ds = (
    train_ds
    .shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=AUTOTUNE)
)

for images in augmented_train_ds:
    try:
        print(images)
    except Exception as e:
        print(e)