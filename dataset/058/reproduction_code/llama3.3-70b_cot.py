# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from keras_cv import models
from keras_cv.models import RetinaNet
from keras_cv.models.object_detection import RetinaNet
from keras_cv.models.object_detection.retinanet import RetinaNet
import numpy as np

# Set up minimal environment
np.random.seed(0)
tf.random.set_seed(0)

# Create a simple dataset for object detection
# For this example, we'll use a dummy dataset with 10 images and 10 bounding boxes
num_images = 10
num_bounding_boxes = 10
image_size = (512, 512)
images = np.random.rand(num_images, image_size[0], image_size[1], 3)
bounding_boxes = np.random.rand(num_images, num_bounding_boxes, 4)
class_labels = np.random.randint(0, 10, size=(num_images, num_bounding_boxes))

# Create a RetinaNet model
model = RetinaNet(
    classes=10,
    bounding_box_format="xyxy",
    image_shape=(image_size[0], image_size[1], 3),
)

# Create a dataset for training
train_ds = tf.data.Dataset.from_tensor_slices((images, bounding_boxes, class_labels))
train_ds = train_ds.batch(1)

# Try to train the model
try:
    model.fit(
        train_ds,
        epochs=1,
    )
except Exception as e:
    print(f"Error: {e}")