import tensorflow as tf
from tensorflow.keras import layers, models

# Minimal model with conv + batch norm to have those variables
def create_model():
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(16, 3, padding='same')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10)(x)
    return tf.keras.Model(inputs, outputs)

# Create model and optimizer
model = create_model()
optimizer = tf.keras.optimizers.Adam()

# Dummy input and dummy labels
x = tf.random.normal((4, 32, 32, 3))
y = tf.random.uniform((4,), maxval=10, dtype=tf.int32)

# Intentionally create a loss NOT connected to model outputs
# For example, constant loss so gradients for model variables do not exist
loss = tf.constant(1.0)

with tf.GradientTape() as tape:
    # Forward pass (ignored)
    _ = model(x, training=True)
    # Use constant loss instead of model output based loss
    # So gradients w.r.t model variables will be None
    loss = tf.constant(1.0)

# Compute gradients
grads = tape.gradient(loss, model.trainable_variables)

# Apply gradients (some will be None)
optimizer.apply_gradients(zip(grads, model.trainable_variables))