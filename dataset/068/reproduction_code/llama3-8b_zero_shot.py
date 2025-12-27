!pip install keras-cv tensorflow
import kerascv as kcv
from typing import List
import numpy as np

class EvaluateCOCOMetricsCallback(kcv.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        pass  # This function should not be called during training

yolo = kcv.YOLOV8(
    input_shape=(640, 640, 3),
    anchors=kcv.yolov8.anchors(),
    classes=1,
)

train_ds = kcv.dataset.KerasCVDataset(
    image_paths=["path/to/train/image"],
    annotations=[{"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2, "class_id": 1}],
    batch_size=16,
)

val_ds = kcv.dataset.KerasCVDataset(
    image_paths=["path/to/validation/image"],
    annotations=[{"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2, "class_id": 1}],
    batch_size=16,
)

yolo.fit(train_ds, validation_data=val_ds, epochs=3, callbacks=[EvaluateCOCOMetricsCallback()])