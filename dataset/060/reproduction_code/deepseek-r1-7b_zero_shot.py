import tensorflow as tf
from tensorflow.keras.applications import preprocess_input
from keras_cv import layers

# Load your dataset with proper dtypes; assuming images are uint8 and masks are int64
train_ds = ...  # Ensure correct dtype for images and masks

def augment_fn(ds):
    image, mask = ds['image'], ds['segmentation_mask']
    # Normalize images to float32 (common for augmentation layers)
    image = tf.cast(image, dtype=tf.float32) / 255.0
    # Ensure masks are int64 or another compatible type for augmentation
    mask = tf.cast(mask, dtype=tf.int64)
    return {'images': image, 'segmentation_masks': mask}

def resize_fn(ds):
    images, masks = ds['images'], ds['segmentation_masks']
    images = layers.resize(images, (128, 128))
    masks = layers.resize(masks, (128, 128), method='nearest')
    return {'images': images, 'segmentation_masks': masks}

augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpack_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

resized_val_ds = (
    val_ds.map(resize_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpack_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)