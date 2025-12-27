import tensorflow as tf
import keras_cv

BATCH_SIZE = 1
AUTOTUNE = tf.data.AUTOTUNE

def augment_fn(sample):
    augment = keras_cv.layers.RandAugment()
    return augment(sample)

def unpackage_inputs(sample):
    return sample["images"], sample["segmentation_masks"]

# Create dummy dataset with segmentation_masks dtype=int64 to trigger the error
def create_dataset():
    images = tf.random.uniform((160, 160, 3), dtype=tf.float32)
    masks = tf.random.uniform((160, 160, 1), maxval=2, dtype=tf.int64)
    sample = {"images": images, "segmentation_masks": masks}
    ds = tf.data.Dataset.from_tensors(sample).repeat()
    return ds

train_ds = create_dataset()
val_ds = create_dataset()

augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=AUTOTUNE)
)

resized_val_ds = (
    val_ds.batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=AUTOTUNE)
)

for _ in augmented_train_ds.take(1):
    pass