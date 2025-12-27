import tensorflow as tf
from efficientdet import EfficientDet

# Load the model
model = EfficientDet("efficientdet_d1_coco17_tpu-32")

# Create an input placeholder
input_image = tf.keras.layers.Input(shape=(1024, 1024, 3))

# Compile the model without providing a loss function
model.compile(optimizer='adam', run_eagerly=True)

# Train the model with no loss function specified
model.fit(input_image, epochs=1)