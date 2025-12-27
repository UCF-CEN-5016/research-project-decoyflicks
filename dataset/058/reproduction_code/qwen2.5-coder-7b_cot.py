import tensorflow as tf
from keras_cv import models
from keras_cv import utils
from typing import List, Tuple

def read_image(image_path: tf.Tensor) -> tf.Tensor:
    """Read and decode a JPEG image from a file path tensor."""
    content = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(content, channels=3)
    return image

def read_label(label_path: tf.Tensor) -> tf.Tensor:
    """Read label content from a file path tensor and flatten."""
    content = tf.io.read_file(label_path)
    labels = tf.reshape(content, [-1])
    return labels

def build_dataset(sample_pairs: List[Tuple[str, str]], batch_size: int = 1) -> tf.data.Dataset:
    """Create a minimal dataset of (image, label) pairs for training."""
    ds = tf.data.Dataset.from_tensor_slices(sample_pairs)
    ds = ds.map(lambda image_path, label_path: (read_image(image_path), read_label(label_path)))
    ds = ds.take(batch_size)
    return ds

# Sample dataset (replace with actual paths and annotations)
sample_pairs = [
    ('images/train/image_000.jpg', 'annotations/train/box_000.txt'),
    ('images/train/image_001.jpg', 'annotations/train/box_001.txt'),
]

batch_size = 1
ds = build_dataset(sample_pairs, batch_size=batch_size)