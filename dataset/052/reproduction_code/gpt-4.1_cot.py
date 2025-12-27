import tensorflow as tf
import numpy as np

# Minimal model that outputs a 4D tensor with shape (batch_size, 4, None, None)
# To simulate the error, we'll create dummy inputs and outputs.

# Define input shape with dynamic dimensions
input_layer = tf.keras.Input(shape=(None,), dtype=tf.int32)

# For simplicity, embed tokens
embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=8)(input_layer)  # (batch, seq_len, 8)

# Add a Conv1D layer with padding='same' to keep dimensions
conv = tf.keras.layers.Conv1D(filters=4, kernel_size=3, padding='same')(embedding)  # (batch, seq_len, 4)

# Intentionally transpose or reshape to get rank 4 tensor (simulate unexpected shape)
# For example, expand dims twice
x = tf.expand_dims(conv, axis=2)  # (batch, seq_len, 1, 4)
x = tf.transpose(x, perm=[0, 3, 1, 2])  # (batch, 4, seq_len, 1)

# Now apply softmax layer (which expects rank 2 or 3 commonly)
softmax = tf.keras.layers.Softmax(axis=-1)
output = softmax(x)  # This may cause the error

model = tf.keras.Model(inputs=input_layer, outputs=output)

# Dummy data
batch_size = 2
seq_len = 5
x_train = np.random.randint(0, 1000, size=(batch_size, seq_len))
y_train = np.random.randint(0, 2, size=(batch_size, 4, seq_len, 1))

# Compile model with categorical crossentropy (expects probs)
model.compile(optimizer='adam', loss='categorical_crossentropy')

try:
    model.fit(x_train, y_train, epochs=1)
except Exception as e:
    print("Error during model.fit:")
    print(e)