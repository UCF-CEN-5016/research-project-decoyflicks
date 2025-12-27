import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.metrics import MeanIoU
from keras_cv.models import YoloV8Detector
from keras_cv import bounding_box
import numpy as np

# Simulate a dataset with varying number of bounding boxes
# Each image has a different number of boxes
labels = [
    {
        'boxes': np.array([[10, 20, 30, 40]]),  # 1 box
        'classes': np.array([[1]]),
        'image_id': np.array([0])
    },
    {
        'boxes': np.array([[50, 60, 70, 80], [90, 100, 110, 120]]),  # 2 boxes
        'classes': np.array([[2], [3]]),
        'image_id': np.array([1])
    }
]

# Convert to a dataset
def preprocess(x, y):
    # Dummy image (replace with actual image data)
    return x, y

dataset = tf.data.Dataset.from_tensor_slices((np.zeros((2, 224, 224, 3)), labels))
dataset = dataset.map(preprocess)

# Create a YoloV8 model
model = YoloV8Detector(input_shape=(224, 224, 3), num_classes=3)

# Compile the model
model.compile(optimizer='adam', loss=MeanAbsoluteError())

# Train the model (this may not be necessary, but to trigger evaluation)
model.fit(dataset, epochs=1, batch_size=1)

# Now, during evaluation, the model may fail due to shape mismatch

predicted_boxes = model.predict(images)
true_boxes = [label['boxes'] for label in labels]
concatenated_boxes = tf.concat([predicted_boxes, true_boxes], axis=0)

max_boxes = 2
padded_labels = []
for label in labels:
    boxes = label['boxes']
    classes = label['classes']
    if len(boxes) < max_boxes:
        pad_boxes = np.zeros((max_boxes - len(boxes), 4))
        pad_classes = np.zeros((max_boxes - len(classes), 1))
        boxes = np.vstack([boxes, pad_boxes])
        classes = np.vstack([classes, pad_classes])
    padded_labels.append({
        'boxes': boxes,
        'classes': classes,
        'image_id': label['image_id']
    })