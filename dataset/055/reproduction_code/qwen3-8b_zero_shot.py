import tensorflow as tf

# Define a custom layer that uses einsum and assumes batch size is 1
class EinsumLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EinsumLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Assume inputs are of shape (1, seq_len, d_k)
        # We use einsum to compute a matrix multiplication
        return tf.einsum('ijk->ij', inputs)

# Create a model with an input shape of (10,)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(10,)),
    tf.keras.layers.Lambda(lambda x: tf.reshape(x, (1, 10, 1))),
    EinsumLayer()
])

# Create input with batch size 64
input_data = tf.random.normal([64, 10])

# Attempt to call the model
try:
    output = model(input_data)
    print("Model output:", output)
except Exception as e:
    print("Error:", e)