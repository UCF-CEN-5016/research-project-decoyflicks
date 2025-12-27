import numpy as np
from keras_cv import applications
import tensorflow as tf

# Load a sample image
img = tf.io.read_file('path_to_your_image.jpg')
img = tf.image.resize(img, (224, 224))

# Define the model and its configuration
model = applications.MobileNetV2(weights='imagenet', include_top=False)
input_shape = (224, 224, 3)

# Fit the model with error
model.fit(np.array([img]), epochs=1, verbose=0)

print("Model fitted successfully!")