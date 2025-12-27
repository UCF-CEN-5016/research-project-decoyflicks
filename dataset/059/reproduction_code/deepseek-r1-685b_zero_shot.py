import tensorflow as tf
from keras_cv.models import YOLOV8Detector

model = YOLOV8Detector(
    num_classes=1,
    bounding_box_format="xywh",
    backbone="yolo_v8_xs_backbone_coco"
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.MeanSquaredError()

model.compile(optimizer=optimizer, loss=loss)

dummy_images = tf.random.uniform((2, 512, 512, 3))
dummy_labels = {
    "boxes": tf.random.uniform((2, 100, 4)),
    "classes": tf.random.uniform((2, 100), maxval=1, dtype=tf.int32)
}

model.fit(dummy_images, dummy_labels, epochs=1)