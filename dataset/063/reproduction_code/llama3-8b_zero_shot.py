```python
import tensorflow as tf

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices(tf.random.uniform((10, 160, 160, 1), minval=0, maxval=255, dtype=tf.float32))
augment_fn = lambda x: tf.image.random_crop(x, size=(150, 150, 1), seed=42)

def unpackage_inputs(x):
    return x

augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

