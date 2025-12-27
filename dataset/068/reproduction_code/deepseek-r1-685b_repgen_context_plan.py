import tensorflow as tf
from keras_cv.models import YOLOV8Detector
from keras_cv.datasets import pascal_voc

# Bug Type: Tensor Shape Mismatch in COCO Metrics
# Bug Description: Error during validation when concatenating boxes with inconsistent shapes
# Reproduces the "ConcatOp: Dimension 1 must be equal" error from the KerasCV YOLOv8 example

# Minimal dataset (using PascalVOC for simplicity)
train_ds = pascal_voc.load(bounding_box_format="xywh", split="train").take(2)
val_ds = pascal_voc.load(bounding_box_format="xywh", split="validation").take(2)

# Simplified YOLOv8 model
yolo = YOLOV8Detector(
    num_classes=20,
    bounding_box_format="xywh",
    backbone="resnet50",
    fpn_depth=1
)

# Custom callback that triggers the error
class BuggyCOCOCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Force metrics computation which triggers the concat error
        self.model.compute_metrics(val_ds, force=True)

# Training setup that reproduces the bug
yolo.compile(
    optimizer="adam",
    classification_loss="binary_crossentropy",
    box_loss="ciou"
)

# This will fail during validation with the shape mismatch
yolo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    callbacks=[BuggyCOCOCallback()],
    verbose=1
)