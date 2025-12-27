import tensorflow as tf
from tensorflow.keras import layers
from keras_cv import models
import numpy as np

def create_mock_dataset():
    images = tf.random.uniform(shape=(2, 300, 300, 3))
    class_ids = tf.constant([[3, 1, 0], [0, 2, 1]])
    bounding_boxes = tf.random.uniform(shape=(2, 3, 4))
    dataset = tf.data.Dataset.from_tensor_slices((images, class_ids, bounding_boxes))
    return dataset

retina_net = models.RetinaNet(
    num_classes=2,
    backbone=models.RetinaNetBackbone('USE_BACKBONE_MODEL'),
    pretrained_backbone=False
)

train_ds = create_mock_dataset()

loss_fn = models.retinanet.RetinaNetLoss(num_classes=2)
optimizer = tf.keras.optimizers.Adam()

retina_net.compile(loss=loss_fn, optimizer=optimizer)

retina_net.fit(
    train_ds,
    epochs=1
)