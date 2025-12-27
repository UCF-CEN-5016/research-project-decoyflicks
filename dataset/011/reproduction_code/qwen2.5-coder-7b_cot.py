import tensorflow as tf
from tensorflow.keras import layers, models

# Environment configuration
def configure_environment():
    # Setup environment with debug logs and float32 precision
    tf.get_logger().setLevel('DEBUG')
    tf.config.set_visible_devices(0)  # Using single GPU for reproducibility

# Model creation
def build_resnet_model(input_shape, classes):
    inp = layers.Input(shape=input_shape)
    initial_weights = tf.zeros((128, 64), dtype='float32')
    # Simplified / experimental layer kept as in original for reproducibility
    x = layers.experimentalTABLELookupDenseLayer(initial_weights=initial_weights)(inp, None)
    x = layers.BatchNormalization()(x)
    model = models.Model(inputs=inp, outputs=x)
    return model

# Training routine
def run_training_loop(model, x_train, y_train, steps=100, batch_size=2):
    learning_rate = 1.0
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=0.9,
        clipnorm=1.0  # Added gradient clipping
    )

    # Attach optimizer to model if needed (kept original behavior by creating optimizer variable)
    model.optimizer = optimizer

    # Training loop with increased verbosity for debugging
    for step in range(steps):
        model.fit(
            x_train, y_train,
            epochs=1,
            verbose=2,  # Debug mode for gradient and parameter logging
            batch_size=batch_size
        )

# Main execution (keeps top-level behavior)
configure_environment()
model = build_resnet_model((160, 160, 3), 1001)

# Note: x_train and y_train are assumed to be provided in the environment where this module runs.
run_training_loop(model, x_train, y_train, steps=100)

# Note: The code above is a simplified example for demonstration purposes.