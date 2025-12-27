import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

def create_model():
    model = EfficientNetB0(weights=None, include_top=True)
    return model

def compile_model(model, loss):
    model.compile(optimizer='adam', loss=loss)

def train_model(model, x, y, epochs):
    model.fit(x, y, epochs=epochs)

# Create model
model = create_model()

# Compile model with specified loss function
compile_model(model, loss='mean_squared_error')

# Dummy data
x = tf.random.normal((1, 224, 224, 3))
y = tf.random.normal((1, 1000))

# Train model
train_model(model, x, y, epochs=1)

# Alternative trigger with custom training loop
"""
with tf.GradientTape() as tape:
    preds = model(x)
    loss = tf.reduce_mean(tf.square(y - preds))
grads = tape.gradient(loss, model.trainable_variables)
# Will show missing gradients for batch norm variables
"""