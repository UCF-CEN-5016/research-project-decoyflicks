import tensorflow as tf

BATCH_SIZE = 2
AUTOTUNE = tf.data.AUTOTUNE

def augment_fn(image, label):
    augmenter = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomTranslation(0.1,0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomCrop(160,160),
    ])
    image = augmenter(image)
    return image, label

def unpackage_inputs(image, label):
    return image, label

dummy_images = tf.random.uniform([BATCH_SIZE * 10, 160, 160, 1], dtype=tf.float32)
dummy_labels = tf.random.uniform([BATCH_SIZE * 10, 160, 160, 1], maxval=2, dtype=tf.int64)

train_ds = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels))

augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=AUTOTUNE)
)

for batch_images, batch_labels in augmented_train_ds.take(1):
    print(batch_images.shape, batch_images.dtype, batch_labels.shape, batch_labels.dtype)