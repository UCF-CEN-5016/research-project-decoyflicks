import tensorflow as tf

# Define a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(10)
])

# Compile the model without a loss function
model.compile(optimizer='adam')

# Sample data
X = tf.random.normal((32, 10))
y = tf.random.normal((32, 10))

# Training loop that produces the warning
for epoch in range(10):
    with tf.GradientTape() as tape:
        outputs = model(X)
        # Note: No loss function is defined
    gradients = tape.gradient(outputs, model.trainable_variables)
    print(f"Epoch {epoch}: Gradients = {gradients}")