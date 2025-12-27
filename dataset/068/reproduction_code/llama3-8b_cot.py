import tensorflow as tf
from keras_cv.metrics import box_coco_metrics

# Set up environment
tf.config.run_functions_eagerly(True)

# Define model and data
model = tf.keras.models.Sequential([...])  # Replace with YOLOV8 model from KerasCV example
train_ds = ...  # Replace with training dataset
val_ds = ...  # Replace with validation dataset

# Set up callbacks
callbacks = [box_coco_metrics.EvaluateCOCOMetricsCallback(val_ds, "model.h5")]

# Train the model for one epoch
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    callbacks=callbacks,
)

# This should trigger the bug!