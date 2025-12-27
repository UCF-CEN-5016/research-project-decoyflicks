import tensorflow as tf
from tensorflow import keras
from keras_cv.layers import RandAugment

def unpackage_inputs(inputs):
    return inputs["images"], inputs["segmentation_masks"]

def augment_fn(inputs):
    inputs["images"] = tf.cast(inputs["images"], tf.float32)
    augmenter = RandAugment(value_range=(0, 255), augmentations_per_image=3)
    return augmenter(inputs)

def resize_fn(inputs):
    inputs["images"] = tf.image.resize(inputs["images"], (160, 160))
    inputs["segmentation_masks"] = tf.image.resize(inputs["segmentation_masks"], (160, 160))
    return inputs

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def create_dummy_dataset():
    images = tf.random.uniform((100, 160, 160, 3), maxval=255, dtype=tf.uint8)
    masks = tf.random.uniform((100, 160, 160, 1), maxval=1, dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices({"images": images, "segmentation_masks": masks})
    return ds

train_ds = create_dummy_dataset()
val_ds = create_dummy_dataset()

augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=AUTOTUNE)
)

resized_val_ds = (
    val_ds.map(resize_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=AUTOTUNE)
)