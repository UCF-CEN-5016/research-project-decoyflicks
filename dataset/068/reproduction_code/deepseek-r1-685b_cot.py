import tensorflow as tf
import keras_cv
from keras_cv.callbacks import PyCOCOCallback

# Create minimal dataset with inconsistent box shapes
images = tf.random.uniform((2, 256, 256, 3))
# First image has 2 boxes, second has 1 box (causing the error)
boxes = [
    tf.constant([[[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]], dtype=tf.float32),  # shape [1,2,4]
    tf.constant([[[0.3, 0.4, 0.5, 0.6]]], dtype=tf.float32)  # shape [1,1,4]
]
classes = [tf.constant([[0, 1]]), tf.constant([[0]])]

# Create dataset
ds = tf.data.Dataset.from_tensor_slices((images, {"boxes": boxes, "classes": classes})).batch(2)

# Minimal YOLOv8 model
yolo = keras_cv.models.YOLOV8Detector(
    num_classes=1,
    bounding_box_format="xywh",
    backbone=keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_s_backbone")
)

# Compile with dummy optimizer
yolo.compile(optimizer="adam", classification_loss="binary_crossentropy")

# This will fail during evaluation due to inconsistent box shapes
yolo.fit(
    ds,
    validation_data=ds,
    epochs=1,
    callbacks=[PyCOCOCallback(ds, bounding_box_format="xywh")]
)