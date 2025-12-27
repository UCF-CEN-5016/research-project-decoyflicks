import tensorflow as tf
from official.modeling.optimization import optimizer_factory
from official.modeling.optimization.configs import optimization_config
import numpy as np

tf.get_logger().setLevel('ERROR')

batch_size = 32
height, width = 512, 512
num_classes = 80

# Create dummy dataset
x_train = np.random.rand(batch_size, height, width, 3).astype(np.float32)
y_train = np.random.randint(0, num_classes, size=(batch_size, num_classes)).astype(np.float32)

# Define model
model = tf.keras.applications.EfficientNetB0(input_shape=(height, width, 3), include_top=True, classes=num_classes)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train model
model.fit(x_train, y_train, epochs=1, validation_split=0.2)

# Monitor for warnings