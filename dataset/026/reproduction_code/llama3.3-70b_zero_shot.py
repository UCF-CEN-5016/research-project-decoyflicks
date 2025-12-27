import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2')

# Load the image
image = np.array(Image.open('image.jpg').resize((640, 480)))

# Convert the image to a tensor
image_tensor = tf.convert_to_tensor(image)

# Create a batch
batch = tf.expand_dims(image_tensor, 0)

# Make predictions
detections = model(batch)

# Extract the detection boxes, classes, and scores
detection_boxes = detections['detection_boxes'][0].numpy()
detection_classes = detections['detection_classes'][0].numpy().astype(np.int64)
detection_scores = detections['detection_scores'][0].numpy()

# Try to visualize the results
try:
    import matplotlib.patches as patches

    fig, ax = plt.subplots()
    ax.imshow(image)

    for i in range(len(detection_scores)):
        if detection_scores[i] > 0.5:
            box = detection_boxes[i]
            rect = patches.Rectangle((box[1]*image.shape[1], box[0]*image.shape[0]), (box[3]-box[1])*image.shape[1], (box[2]-box[0])*image.shape[0], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    plt.show()
except Exception as e:
    print(f"Error: {e}")