import tensorflow as tf
from tensorflow import keras
from keras_cv import layers as kcv_layers

# Dummy dataset with inconsistent mask types
def create_dummy_dataset():
    images = tf.random.uniform(shape=(160, 160, 3), dtype=tf.float32)
    masks = tf.random.uniform(shape=(160, 160, 1), dtype=tf.int64)
    return tf.data.Dataset.from_tensor_slices((images, masks))

# Augmentation functions with type mismatches
def augment_float_mask(images, masks):
    return {
        'images': images,
        'segmentation_masks': tf.cast(masks, tf.float32)
    }

def augment_int_mask(images, masks):
    return {
        'images': images,
        'segmentation_masks': masks
    }

# Fixed RandomChoice layer with consistent output types
def random_choice(images, masks):
    return tf.cond(tf.random.uniform([]) < 0.5,
                   lambda: augment_float_mask(images, masks),
                   lambda: augment_int_mask(images, masks))

# Reproducing the error
dataset = create_dummy_dataset()

# Apply fixed random_choice function to the dataset
dataset = dataset.map(random_choice)

# This will trigger the type mismatch error
for batch in dataset.take(1):
    print(batch)