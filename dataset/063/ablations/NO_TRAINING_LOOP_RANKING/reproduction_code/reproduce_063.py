import tensorflow as tf
import keras_cv
from keras_cv import layers as cv_layers

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def augment_fn(image):
    image = cv_layers.RandomFlip(mode="horizontal")(image)
    return image

def unpackage_inputs(inputs):
    return inputs

train_ds = tf.data.Dataset.from_tensor_slices(tf.random.uniform((100, 160, 160, 1), dtype=tf.float32))
train_ds = train_ds.shuffle(BATCH_SIZE * 2)
train_ds = train_ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.map(unpackage_inputs)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

try:
    augmented_train_ds = train_ds
except Exception as e:
    print(e)