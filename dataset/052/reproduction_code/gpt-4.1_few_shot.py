import tensorflow as tf
import numpy as np

# Minimal Transformer block with incorrect boolean check causing the error
class FaultySoftmax(tf.keras.layers.Layer):
    def call(self, inputs):
        # Incorrect: using a tf.Tensor in Python boolean context triggers error in graph mode
        if inputs.shape[2] is None:  # This shape dimension is symbolic (None)
            # trying to branch on symbolic shape dimension causes OperatorNotAllowedInGraphError
            return tf.nn.softmax(inputs)
        else:
            return inputs

# Simple model that uses the faulty softmax layer
inputs = tf.keras.Input(shape=(None, 10))  # Sequence length unknown (None)
x = tf.keras.layers.Dense(10)(inputs)
x = FaultySoftmax()(x)  # This layer triggers the error during fit
outputs = tf.keras.layers.Dense(5)(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Dummy dataset
X = np.random.rand(32, 4, 10).astype(np.float32)
y = np.random.randint(0, 5, size=(32, 4))

# This will raise OperatorNotAllowedInGraphError during model.fit()
model.fit(X, y, epochs=1)