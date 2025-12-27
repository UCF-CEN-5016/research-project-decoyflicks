import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# Load the model
model = tf.saved_model.load('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2')

# Load the image
image_path = 'image.jpg'
image = load_img(image_path)
image_np = np.array(image)

# Convert the image to RGB
image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

# Expand the dimensions of the image
image_np_expanded = np.expand_dims(image_np, axis=0)

# Run the model
outputs = model.signatures['default'](tf.convert_to_tensor(image_np_expanded))

# Get the detection boxes, classes, and scores
detection_boxes = outputs['detection_boxes'].numpy()
detection_classes = outputs['detection_classes'].numpy()
detection_scores = outputs['detection_scores'].numpy()

# Visualize the results
fig, ax = plt.subplots()
ax.imshow(image_np)

# Draw the detection boxes
for i in range(detection_boxes.shape[1]):
    box = detection_boxes[0, i, :]
    score = detection_scores[0, i]
    class_id = detection_classes[0, i]

    if score > 0.5:
        # This line triggers the bug
        detection_masks_reframed = tf.image.crop_and_resize(tf.ones((1, 10, 10, 1)), [box], [10, 10])
        rect = patches.Rectangle((box[1] * image_np.shape[1], box[0] * image_np.shape[0]), (box[3] - box[1]) * image_np.shape[1], (box[2] - box[0]) * image_np.shape[0], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[1] * image_np.shape[1], box[0] * image_np.shape[0], f'Class {class_id}, Score {score:.2f}')

plt.show()