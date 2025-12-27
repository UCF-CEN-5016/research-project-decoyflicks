import tensorflow as tf
import numpy as np

def reframe_box_masks_to_image_masks(box_masks, boxes, image_height, image_width):
    num_boxes = tf.shape(box_masks)[0]
    return tf.image.crop_and_resize(
        box_masks,
        boxes,
        box_ind=tf.range(num_boxes),
        crop_size=[image_height, image_width],
        extrapolation_value=0.0)

box_masks = tf.random.uniform([2, 5, 5], dtype=tf.float32)
boxes = tf.constant([[0.1, 0.1, 0.5, 0.5],
                     [0.2, 0.2, 0.6, 0.6]], dtype=tf.float32)
image_height, image_width = 10, 10

reframe_box_masks_to_image_masks(box_masks, boxes, image_height, image_width)