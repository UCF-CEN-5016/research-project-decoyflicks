import tensorflow as tf
from orbit import SingleTaskTrainer, utils

# Set up random seed for reproducibility
tf.random.set_seed(42)

# Define model and dataset parameters
batch_size = 32
image_height, image_width = 128, 128

# Create random uniform input data with shape (batch_size, height, width, 3)
input_data = tf.random.uniform((batch_size, image_height, image_width, 3), minval=0, maxval=255, dtype=tf.float32)

# Define a simple sequential model with one dense layer
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(image_height, image_width, 3)),
    tf.keras.layers.Dense(10)
])

# Compile the model with SGD optimizer, mean_squared_error loss, and accuracy metric
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

# Define labels (random for demonstration)
labels = tf.random.uniform((batch_size,), minval=0, maxval=10, dtype=tf.int32)

# Create a dataset using TensorFlow Dataset API with the generated input data and labels
dataset = tf.data.Dataset.from_tensor_slices((input_data, labels)).batch(batch_size)

# Set up SingleTaskTrainer
label_key = 'labels'
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD()
metrics = [tf.keras.metrics.Accuracy()]

trainer_options = utils.StandardTrainerOptions(train_steps=10)
trainer = SingleTaskTrainer(
    train_dataset=dataset,
    label_key=label_key,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    metrics=metrics,
    trainer_options=trainer_options
)

# Train the model for 10 steps using the trainer's train method
results = trainer.train()

# Verify that the calculated loss contains NaN values
assert tf.math.reduce_any(tf.math.is_nan(results['training_loss'])), "Loss is NaN"

# Monitor GPU memory usage during training
print("GPU Memory Usage:", tf.config.experimental.get_memory_info('GPU')['total'])