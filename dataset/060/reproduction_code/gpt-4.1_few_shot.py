import tensorflow as tf
import tensorflow_datasets as tfds
import keras_cv

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 4
IMG_SIZE = 160

# Load Oxford Pets dataset (image segmentation)
dataset, info = tfds.load("oxford_iiit_pet:3.*.*", with_info=True)
train_ds = dataset["train"].map(lambda x: {
    "images": tf.image.resize(tf.cast(x["image"], tf.float32) / 255.0, [IMG_SIZE, IMG_SIZE]),
    "segmentation_masks": tf.image.resize(tf.cast(x["segmentation_mask"], tf.int64), [IMG_SIZE, IMG_SIZE], method="nearest")
})

# Define a dummy augmentation function that applies RandAugment layer
rand_augment = keras_cv.layers.RandAugment(value_range=(0, 1))

def augment_fn(inputs):
    return rand_augment(inputs)

# This line triggers the TypeError due to inconsistent mask dtypes in augmentation branches
augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
)

for batch in augmented_train_ds.take(1):
    print(batch)