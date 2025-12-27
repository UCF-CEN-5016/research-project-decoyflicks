import tensorflow as tf
import numpy as np
from official.modeling.optimization import optimizer_factory
from official.modeling.optimization.configs import optimization_config

tf.get_logger().setLevel('ERROR')

batch_size = 32
height, width = 512, 512
num_classes = 80

# Create dummy dataset
x_train = np.random.rand(batch_size, height, width, 3).astype(np.float32)
y_train = np.random.randint(0, num_classes, size=(batch_size, num_classes)).astype(np.float32)

# Define model
base_model = tf.keras.applications.EfficientNetB0(input_shape=(height, width, 3), include_top=False, weights='imagenet')
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile model
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model
model.fit(x_train, y_train, epochs=10, validation_split=0.2)