import numpy as np
import tensorflow as tf
import keras_cv

images = np.random.rand(2, 512, 512, 3).astype(np.float32)
boxes = np.array([[[0.1, 0.1, 0.5, 0.5], [0.3, 0.3, 0.8, 0.8]], 
                 [[0.2, 0.2, 0.6, 0.6], [0.4, 0.4, 0.9, 0.9]]], dtype=np.float32)
classes = np.array([[1, 2], [1, 2]], dtype=np.float32)

train_ds = tf.data.Dataset.from_tensor_slices((images, {"boxes": boxes, "classes": classes})).batch(2)

model = keras_cv.models.RetinaNet(
    classes=2,
    bounding_box_format="xywh",
    backbone="resnet50"
)

model.compile(
    optimizer="adam",
    classification_loss="focal",
    box_loss="smoothl1",
)

model.fit(train_ds, epochs=1)