import tensorflow as tf
from keras_cv.models import RetinaNet
from keras_cv.losses import RetinaNetLoss

# 1. Create a synthetic dataset with invalid labels (label > num_classes)
batch_size = 2
images = tf.random.uniform((batch_size, 512, 512, 3))
# Invalid labels: 81 is out of bounds for num_classes=80
boxes = tf.constant([[[0.1, 0.2, 0.3, 0.4]], [[0.5, 0.6, 0.7, 0.8]]], dtype=tf.float32)
labels = tf.constant([[81], [81]], dtype=tf.int32)  # Invalid labels!

# 2. Initialize RetinaNet (COCO defaults)
num_classes = 80  # COCO classes
model = RetinaNet(
    num_classes=num_classes,
    bounding_box_format="xywh",
    backbone="resnet50"
)

# 3. Compile with RetinaNetLoss (triggers GatherV2 during label encoding)
model.compile(
    optimizer="adam",
    loss=RetinaNetLoss(num_classes=num_classes)
)

# 4. Train (will fail with "indices[0,81] = 81 is not in [0, 80]")
try:
    model.fit(x=images, y={"boxes": boxes, "labels": labels}, epochs=1)
except Exception as e:
    print(f"Error: {e}")  # Expected: "indices[...] is not in [0, 80]"