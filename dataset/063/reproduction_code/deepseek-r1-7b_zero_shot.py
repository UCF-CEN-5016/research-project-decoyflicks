import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def augment_fn(image, label):
    # Random resizing within 80-120% of original size
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    new_h = tf.random.uniform([h*0.8, h*1.2], dtype=tf.float32)
    new_w = tf.random.uniform([w*0.8, w*1.2], dtype=tf.float32)
    
    # Random rotation between -15 to 15 degrees
    angle = tf.random.uniform([], -15, 15, dtype=tf.float32)
    
    # Apply random brightness and contrast adjustments (float operations)
    brightness_delta = tf.random.uniform([], -0.1, 0.1, dtype=tf.float32)
    image = tf.image.adjust_brightness(image, delta=brightness_delta)
    contrast_delta = tf.random.uniform([], -0.1, 0.1, dtype=tf.float32)
    image = tf.image.adjust_contrast(image, delta=contrast_delta)
    
    # Convert images and labels to appropriate dtypes
    image = tf.cast(image, dtype=tf.float32) / 255.0
    label = tf.cast(label, dtype=tf.int64)
    
    return (image, label)

def unpack_input(element):
    image, label = element
    return tf.cast(image, dtype=tf.float32), tf.cast(label, dtype=tf.int64)

# Creating the dataset
batch_size = 32

train_ds = tf.data DirectoryDataset("path/to/training/images")
train_ds = train_ds.map(augment_fn)
train_ds = train_ds.shuffle(buffer_size=batch_size * 2)
train_ds = train_ds.batch(batch_size)
train_ds = train_ds prefetch(buffer_size=tf.data.AUTOTUNE)

augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpack_input)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Note: Ensure that all data sources are correctly handled and paths are accurate