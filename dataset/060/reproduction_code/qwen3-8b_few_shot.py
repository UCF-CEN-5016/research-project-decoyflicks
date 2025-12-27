import tensorflow as tf
import keras
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

# Problematic RandomChoice layer with inconsistent output types
random_choice = kcv_layers.RandomChoice(
    [
        augment_float_mask,
        augment_int_mask
    ],
    num_classes=2
)

# Reproducing the error
dataset = create_dummy_dataset()

# This will trigger the type mismatch error
for batch in dataset.map(random_choice).take(1):
    print(batch)