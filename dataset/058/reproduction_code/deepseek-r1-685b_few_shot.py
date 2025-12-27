import tensorflow as tf
import keras_cv
import numpy as np

# Create a minimal dataset with invalid bounding boxes
images = tf.random.uniform((2, 512, 512, 3))  # 2 sample images
# Problematic boxes: class_id=0 (background) with invalid coordinates
boxes = tf.constant([
    [[0, 0, 0, 0, 0]],  # Background box with zero coordinates
    [[1, -10, -10, 600, 600]]  # Box outside image bounds
], dtype=tf.float32)

# Create dataset
ds = tf.data.Dataset.from_tensor_slices((images, boxes))
ds = ds.batch(2)

# Create RetinaNet model
model = keras_cv.models.RetinaNet(
    classes=1,  # Just one class (plus background)
    bounding_box_format="xyxy"
)

# Attempt training - will fail during label encoding
model.compile(
    optimizer="adam",
    classification_loss="focal",
    box_loss="smoothl1"
)

# This will raise the InvalidArgumentError
model.fit(ds, epochs=1)