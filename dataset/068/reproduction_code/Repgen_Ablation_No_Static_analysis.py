import tensorflow as tf
from keras_cv.models import YOLOV8Backbone, YOLOV8Detector
from keras_cv.metrics import BoxCOCOMetrics
from keras.utils.data_utils import Sequence
import numpy as np

# Define batch size and image dimensions
batch_size = 32
image_height, image_width = 640, 640

# Create random uniform input data with shape (batch_size, height, width, 3)
data = np.random.uniform(0, 255, (batch_size, image_height, image_width, 3))

# Load pre-trained COCO weights for YOLOV8 backbone
backbone = YOLOV8Backbone.from_preset("yolov8_s_coco")

# Initialize the YOLOV8Detector model
model = YOLOV8Detector(
    num_classes=number_of_classes,
    bounding_box_format="xyxy",
    fpn_depth=1,
    backbone=backbone
)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, global_clipnorm=5.0),
    loss={"class": "binary_crossentropy", "box": "ciou"},
    metrics=["accuracy"]
)

# Create a validation dataset (assuming val_ds is defined elsewhere)
val_ds = Sequence.from_generator(lambda: data, output_signature=(tf.TensorSpec((None, image_height, image_width, 3), tf.float32)))

# Set up EvaluateCOCOMetricsCallback
class EvaluateCOCOMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, data, save_path):
        super().__init__()
        self.data = data
        self.metrics = BoxCOCOMetrics(bounding_box_format="xyxy", evaluate_freq=1e9)
        self.save_path = save_path
        self.best_map = -1.0

    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for batch in self.data:
            images, y_true = batch[0], batch[1]
            y_pred = model.predict(images, verbose=0)
            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)
        current_map = metrics["MaP"]
        if current_map > self.best_map:
            self.best_map = current_map
            model.save(self.save_path)

# Train the model
model.fit(
    train_ds,
    epochs=3,
    validation_data=val_ds,
    callbacks=[EvaluateCOCOMetricsCallback(val_ds, 'model.h5')]
)

# Visualize detections
def visualize_detections(model, dataset, bounding_box_format):
    images, y_true = next(iter(dataset.take(1)))
    y_pred = model.predict(images)
    y_pred = tf.ragged.stack(y_pred)
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=2,
        cols=2,
        show=True,
        font_scale=0.7,
        class_mapping=class_mapping
    )

visualize_detections(model, val_ds, "xyxy")