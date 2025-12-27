import tensorflow as tf
from tensorflow import keras
from keras_cv import layers as kcv_layers
from typing import Dict

def build_dummy_dataset() -> tf.data.Dataset:
    """Create a dummy dataset with inconsistent mask dtypes."""
    images = tf.random.uniform(shape=(160, 160, 3), dtype=tf.float32)
    masks = tf.random.uniform(shape=(160, 160, 1), dtype=tf.int64)
    return tf.data.Dataset.from_tensor_slices((images, masks))

def to_float_mask(images: tf.Tensor, masks: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Return a sample with the mask cast to float32."""
    return {
        'images': images,
        'segmentation_masks': tf.cast(masks, tf.float32)
    }

def to_int_mask(images: tf.Tensor, masks: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Return a sample with the mask left as an integer dtype."""
    return {
        'images': images,
        'segmentation_masks': masks
    }

def random_choice_aug(images: tf.Tensor, masks: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Randomly choose between returning a float or int mask."""
    return tf.cond(
        tf.random.uniform([]) < 0.5,
        lambda: to_float_mask(images, masks),
        lambda: to_int_mask(images, masks)
    )

if __name__ == "__main__":
    dataset = build_dummy_dataset()
    dataset = dataset.map(random_choice_aug)

    for sample in dataset.take(1):
        print(sample)