import tensorflow as tf

# Define a simple sequential model with a Dense layer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(784,))
])

# Create a synthetic training dataset with random data
train_dataset = tf.data.Dataset.from_tensor_slices((tf.random.normal([32, 784]), tf.random.uniform([32], maxval=10, dtype=tf.int32)))
train_dataset = train_dataset.batch(32)

# Set batch size to 32 and define number of steps per epoch
batch_size = 32
steps_per_epoch = len(train_dataset)

# Compile the model using SGD optimizer, categorical crossentropy loss, and accuracy metrics
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on the synthetic dataset for one epoch
try:
    history = model.fit(train_dataset, epochs=1)
except tf.errors.InvalidArgumentError as e:
    if "NaN" in str(e):
        print("Loss became NaN during training")
    else:
        raise e