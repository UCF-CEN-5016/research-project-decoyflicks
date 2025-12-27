import tensorflow as tf
import keras
from keras_cv import models, bounding_box
import numpy as np

def dataset():
    images = tf.constant([[[1.0, 1.0, 1.0, 1.0]]])
    boxes = tf.constant([[0.0, 0.0, 640.0, 640.0]])
    classes = tf.constant([[10]])
    annotations = {'boxes': boxes, 'classes': classes}
    return tf.data.Dataset.from_tensor_slices((images, annotations))

model = models.RetinaNet(
    bounding_box_format='yxyx',
    num_classes=10,
    backbone=keras_cv.models.RetinaNetBackbone(
        backbone='resnet50',
        input_shape=(640, 640, 3)
    )
)

model.compile(optimizer='adam', loss='mse')
model.fit(dataset(), epochs=1)