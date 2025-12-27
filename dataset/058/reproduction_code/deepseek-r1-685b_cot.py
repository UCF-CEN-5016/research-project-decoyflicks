import tensorflow as tf
import keras_cv
from keras_cv.models import RetinaNet
from keras_cv import bounding_box

# 1. Create a minimal dataset with potential invalid entries
def create_broken_dataset():
    # Sample with invalid class_id (1 when we only have 1 class)
    images = tf.random.uniform((2, 512, 512, 3))
    boxes = tf.constant([
        [[0.1, 0.1, 0.5, 0.5, 1], [0.6, 0.6, 0.8, 0.8, 1]],  # Two valid boxes
        [[0.0, 0.0, 0.0, 0.0, -1]]  # Invalid box (class_id = -1)
    ], dtype=tf.float32)
    return tf.data.Dataset.from_tensor_slices((images, boxes))

# 2. Set up model with minimal config
num_classes = 1  # Only class 0 should exist
model = RetinaNet(
    classes=num_classes,
    bounding_box_format="xywh",
    backbone=keras_cv.models.ResNet50Backbone.from_preset("resnet50_imagenet")
)

# 3. Try to train - this will trigger the error
train_ds = create_broken_dataset().batch(2)
model.compile(
    optimizer="adam",
    classification_loss="focal",
    box_loss="smoothl1",
)

# This will raise the InvalidArgumentError
model.fit(train_ds, epochs=1)

def filter_invalid_boxes(images, boxes):
    boxes = bounding_box.validate(boxes)
    return images, boxes

train_ds = create_broken_dataset().map(filter_invalid_boxes).batch(2)