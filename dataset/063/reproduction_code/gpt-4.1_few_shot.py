import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 4

def augment_fn(image, label):
    # Randomly choose between two augmentations with different output types
    def aug_float():
        return tf.cast(image, tf.float32), label
    def aug_int():
        return tf.cast(image, tf.int64), label

    choice = tf.random.uniform([], maxval=2, dtype=tf.int32)
    return tf.switch_case(choice, branch_fns={0: aug_float, 1: aug_int})

def unpackage_inputs(image, label):
    return image, label

# Create example dataset with images as int32 (simulate labels or images)
images = tf.random.uniform([BATCH_SIZE*2, 160, 160, 1], maxval=255, dtype=tf.int32)
labels = tf.zeros([BATCH_SIZE*2], dtype=tf.int32)
train_ds = tf.data.Dataset.from_tensor_slices((images, labels))

# Dataset pipeline triggering the dtype conflict error
augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=AUTOTUNE)
)

for batch in augmented_train_ds.take(1):
    print(batch)