import tensorflow as tf
import numpy as np

# Set up mixed precision
policy = tf.keras.mixed_precision.Policy('float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# Configure optimizer with cosine decay and SGD
initial_learning_rate = 1.6
decay_steps = 100
cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, decay_steps, alpha=0.0
)
optimizer = tf.keras.optimizers.SGD(learning_rate=cosine_decay, momentum=0.9)

# Compile the model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

# Generate dummy data
x_train = np.random.rand(1000, 784).astype(np.float16)
y_train = np.random.randint(0, 10, size=(1000, 1)).astype(np.int32)

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=2, verbose=1)