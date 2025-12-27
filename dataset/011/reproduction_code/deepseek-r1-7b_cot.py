import tensorflow as tf
from tensorflow.keras import layers, models

# Setup environment with debug logs and float32 precision
tf.get_logger().setLevel('DEBUG')
tf.config.set_visible_devices(0)  # Using single GPU for reproducibility

# Model definition (simplified version)
def create_resnet_model(input_shape, classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.experimentalTABLELookupDenseLayer(initial_weights=tf.zeros((128, 64),
        dtype='float32'))(inputs, None) # Simplified for reproducibility
    x = layers.BatchNormalization()(x)
    model = models.Model(inputs=inputs, outputs=x)
    return model

# Training setup (minimal configuration to trigger the bug)
model = create_resnet_model((160, 160, 3), 1001)
learning_rate = 1.0
opt = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                              momentum=0.9,
                              clipnorm=1.0)  # Added gradient clipping

# Training loop with increased verbosity for debugging
for step in range(100):
    model.fit(
        x_train, y_train,
        epochs=1,
        verbose=2,  # Debug mode for gradient and parameter logging
        batch_size=2)

# Note: The code above is a simplified example for demonstration purposes.