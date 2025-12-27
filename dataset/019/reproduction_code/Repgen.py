import tensorflow as tf
from official.tn_expand_condense import TNExpandCondense
import numpy as np  # Add this line to define 'np' for array operations

# Set up environment
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()

# Define model parameters
project_multiple = 2
use_bias = True
activation = 'relu'
input_shape = (100, 768)

# Create random input data
input_data = tf.constant(tf.random.uniform(shape=input_shape, minval=0, maxval=10, dtype=tf.int32), dtype=tf.float32)

# Define model
model = TNExpandCondense(project_multiple=project_multiple, use_bias=use_bias, activation=activation)
output = model(input_data)

# Compile and train model
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.BinaryCrossentropy()

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        predictions = output
        loss = loss_fn(tf.zeros_like(predictions), predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

for epoch in range(5):
    train_step()

# Verify weight changes and layer shape
assert model.weights[0].shape == (input_shape[-1], input_shape[-1] * project_multiple)

# Attempt to create an incorrect-sized model and verify that it raises an AssertionError
try:
    TNExpandCondense(project_multiple=2, use_bias=True, activation='relu', input_shape=(100, 769))
except AssertionError as e:
    print(e)

# Serialize and deserialize model
model_config = model.get_config()
new_layer = TNExpandCondense.from_config(model_config)
assert new_layer.weights[0].shape == (input_shape[-1], input_shape[-1] * project_multiple)

# Save and load model
temp_dir = '/tmp/temp_model'
tf.saved_model.save(model, temp_dir)
loaded_model = tf.saved_model.load(temp_dir)

# Compare predictions
predictions_original = output.numpy()
predictions_loaded = loaded_model(input_data).numpy()
assert np.array_equal(predictions_original, predictions_loaded)