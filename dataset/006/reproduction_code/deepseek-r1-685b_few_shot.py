import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

# Create a model with batch norm layers (common source of this warning)
model = EfficientNetB0(weights=None, include_top=True)

# Compile without specifying loss (or with incorrect setup)
model.compile(optimizer='adam')  # Missing loss argument

# Dummy data
x = tf.random.normal((1, 224, 224, 3))
y = tf.random.normal((1, 1000))

# Triggers the warning
model.fit(x, y, epochs=1)

# Alternative trigger with custom training loop
"""
with tf.GradientTape() as tape:
    preds = model(x)
    loss = tf.reduce_mean(tf.square(y - preds))
grads = tape.gradient(loss, model.trainable_variables)
# Will show missing gradients for batch norm variables
"""