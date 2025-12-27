import os
import tensorflow as tf
from keras_cv.models import YOLOV8Small
from keras_cv.metrics import BoxCOCOMetrics

# Define batch size and image dimensions
batch_size = 32
img_height, img_width = 640, 640

# Load the YOLOV8Small backbone with COCO pre-trained weights
backbone = YOLOV8Small(pretrained_weights="coco")

# Create a dataset using the loaded backbone
train_dataset = ...
val_dataset = ...

# Initialize an instance of YOLOV8Detector
model = YOLOV8Detector(
    backbone=backbone,
    num_classes=len(class_mapping),
    bounding_box_format="xyxy",
    fpn_depth=1
)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss={"classification": "binary_crossentropy", "bounding_box": "ciou"})

# Create a callback to evaluate COCO metrics
coco_metrics = EvaluateCOCOMetricsCallback(val_dataset, "model.h5")

# Train the model
model.fit(train_dataset,
          validation_data=val_dataset,
          epochs=3,
          callbacks=[coco_metrics])

# Assert that the trained model saves the best version based on improving mAP score
assert os.path.exists("model.h5")