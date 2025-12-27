import tensorflow as tf

# Load pre-trained model
model = tf.saved_model.load("faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8")

# Set up input tensor
input_tensor = tf.random.normal([1, 640, 640, 3])

# Attempt to make predictions
outputs = model(input_tensor)

print(outputs)