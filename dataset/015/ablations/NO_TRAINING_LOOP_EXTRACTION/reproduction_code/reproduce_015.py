import numpy as np
import tensorflow as tf

EPSILON = 1e-8

def normalize_boxes(boxes, image_shape):
    if boxes.shape[-1] != 4:
        raise ValueError('boxes.shape[-1] is {:d}, but must be 4.'.format(boxes.shape[-1]))

    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
        height, width = image_shape
    else:
        image_shape = tf.cast(image_shape, dtype=boxes.dtype)
        height = image_shape[..., 0:1]
        width = image_shape[..., 1:2]

    ymin = boxes[..., 0:1] / height
    xmin = boxes[..., 1:2] / width
    ymax = boxes[..., 2:3] / height
    xmax = boxes[..., 3:4] / width

    normalized_boxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
    return normalized_boxes

def show_batch(num_of_examples):
    batch_size = 2
    image_shape = (256, 256)
    boxes = np.random.rand(batch_size, 4).astype(np.float32)
    normalized_boxes = normalize_boxes(boxes, image_shape)
    return normalized_boxes

output = show_batch(num_of_examples=5)
print(output)