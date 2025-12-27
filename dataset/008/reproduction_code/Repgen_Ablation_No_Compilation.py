import tensorflow as tf
from official.vision.detection.configs import faster_rcnn_resnet_v2_640x640_config

# Set TensorFlow version and Python version
%tensorflow_version 2.13.0
!python --version

# Create random input image tensor
input_tensor = tf.random.uniform((1, 640, 640, 3), dtype=tf.float32)

# Load Faster R-CNN Inception ResNet V2 model configuration
model_config = faster_rcnn_resnet_v2_640x640_config.FasterRCNNResNetV2640x640()
model_builder = model_config.model_builder
model = model_builder.build_model()

# Compile the model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss_function)

# Generate random label tensor
label_tensor = tf.random.uniform((1, 1), maxval=model_config.num_classes, dtype=tf.int32)

# Train the model for a single step
with tf.GradientTape() as tape:
    predictions = model(input_tensor)
    loss = loss_function(label_tensor, predictions)

# Verify that the loss is finite and does not contain NaN values
assert tf.math.is_finite(loss), "Loss contains NaN values"

# Monitor GPU memory usage
!nvidia-smi

# Assert GPU memory usage exceeds a predefined threshold (e.g., 1GB)
gpu_memory_usage = float(!nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d '[:space:]')
assert gpu_memory_usage > 1024, "GPU memory usage does not exceed the predefined threshold"