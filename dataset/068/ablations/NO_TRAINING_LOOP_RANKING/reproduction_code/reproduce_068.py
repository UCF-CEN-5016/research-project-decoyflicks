import os
import tensorflow as tf
from tensorflow import keras
import keras_cv
import gdown
import matplotlib.pyplot as plt
from zipfile import ZipFile  # Fixed the undefined variable issue
from functools import partial

tf.random.set_seed(42)

batch_size = 16
image_size = (640, 640)

url = "https://your_coco_dataset_link"
output = "coco_dataset.zip"
gdown.download(url, output, quiet=True)

os.makedirs("coco_dataset", exist_ok=True)
with ZipFile(output, "r") as zipobj:
    zipobj.extractall("coco_dataset")

train_ds = keras.utils.image_dataset_from_directory(
    "coco_dataset/train", label_mode='int', image_size=image_size, batch_size=batch_size
)

val_ds = keras.utils.image_dataset_from_directory(
    "coco_dataset/val", label_mode='int', image_size=image_size, batch_size=batch_size
)

yolo = keras_cv.models.YOLOv8(input_shape=image_size + (3,), classes=80)

yolo.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy')

class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Custom evaluation logic here
        pass

# The following line is where the bug is reproduced
yolo.fit(train_ds, validation_data=val_ds, epochs=3, callbacks=[EvaluateCOCOMetricsCallback()])