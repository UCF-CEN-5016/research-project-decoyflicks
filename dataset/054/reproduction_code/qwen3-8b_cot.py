import tensorflow as tf
from tensorflow.keras import layers, models

# Create dummy data
train_data = tf.random.uniform(shape=(1000, 28, 28))  # 1000 training samples
test_data = tf.random.uniform(shape=(200, 28, 28))    # 200 test samples

# Convert to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data)

# Define a simple model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model with test data as validation
model.fit(train_dataset, epochs=5, validation_data=test_dataset)

model.fit(train_dataset, epochs=5, validation_split=0.2)

val_dataset = train_dataset.take(100)  # Take first 100 samples as validation
train_dataset = train_dataset.skip(100)
model.fit(train_dataset, epochs=5, validation_data=val_dataset)