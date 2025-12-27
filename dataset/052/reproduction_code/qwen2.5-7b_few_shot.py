import tensorflow as tf

# Define a model with a Conv2D layer followed by a GlobalAveragePooling2D and Softmax
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)),  # 2D output
    tf.keras.layers.GlobalAveragePooling2D(),  # Convert to 1D tensor
    tf.keras.layers.Softmax()  # Apply Softmax to the 1D tensor
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Create dummy input data (4D shape)
x_train = tf.random.normal([100, 28, 28, 1])
y_train = tf.random.uniform([100], minval=0, maxval=10, dtype=tf.int32)

# Attempt to train the model
model.fit(x_train, y_train, epochs=1)