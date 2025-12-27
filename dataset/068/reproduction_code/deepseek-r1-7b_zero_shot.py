import os
import tensorflow as tf
from keras_cv import YOLOV8
import numpy as np

# Set up the model
model = YOLOV8('yolov8n')
model.load_weights('https://github.com/keras-ai/YOLoV8/releases/download/v1.0/yolov8n.h5')

# Create a simple dataset with multiple images (batch_size=2)
dataset = tf.data.Dataset.from_tensor slices([
    ('path/to/training/image_1.jpg', [4, 640, 640, 3]),
    ('path/to/training/image_2.jpg', [4, 640, 640, 3]),
]).map(lambda x: model.predict(x[0]))

# Try to concatenate the predictions
batched_predictions = list(dataset.as_numpy_iterator())
# This will trigger the error when trying to concatenate across multiple images in a batch

# Debugging tip - add print statements:
for preds in batched_predictions:
    print(preds.keys())  # Should show dictionary keys: ['boxes', 'box_confidences', 'classes', 'class_probabilities']
    print(preds['boxes'].shape)  # Check box shapes

# Modified Debugging code:
for preds in batched_predictions:
    print("Predictions shape:", next(iter(preds.values())).shape)