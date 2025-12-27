import tensorflow as tf
from official import build_classification_model

# Define constants
BATCH_SIZE = 32
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
NUM_CLASSES = 1000
LEARNING_RATE = 0.001

# Create random input data
input_data = tf.random.uniform((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.float32)
labels = tf.random.uniform((BATCH_SIZE,), maxval=NUM_CLASSES, dtype=tf.int64)

# Build the model
model = build_classification_model(num_classes=NUM_CLASSES)

# Initialize optimizer and metrics
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
top_k_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)

# Forward pass and compute gradients
with tf.GradientTape() as tape:
    predictions = model(input_data, training=True)
    loss = loss_fn(labels, predictions)

# Backward pass
gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Monitor loss values
print("Loss:", loss.numpy())
assert not tf.math.is_nan(loss) and not tf.math.is_inf(loss), "Loss is NaN or Inf"

# Ensure GPU memory usage is within expected limits
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Run multiple epochs of training
for epoch in range(5):
    with tf.GradientTape() as tape:
        predictions = model(input_data, training=True)
        loss = loss_fn(labels, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    accuracy.update_state(labels, predictions)
    top_k_accuracy.update_state(labels, predictions)
    
    print(f"Epoch {epoch+1}, Loss: {loss.numpy()}, Accuracy: {accuracy.result().numpy()}, Top-5 Accuracy: {top_k_accuracy.result().numpy()}")