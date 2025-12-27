import tensorflow as tf
from tensorflow import keras

# Set up a simple point cloud dataset (just for demonstration purposes)
train_points = [[1, 2, 3], [4, 5, 6]]
test_points = [[7, 8, 9], [10, 11, 12]]

# Create a Keras model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    keras.layers.Dense(1)
])

# Split data into training and testing sets
train_dataset = tf.data.Dataset.from_tensor_slices(train_points)
test_dataset = tf.data.Dataset.from_tensor_slices(test_points)

# Train the model using test set as validation set (buggy behavior!)
model.fit(train_dataset.batch(2), epochs=20, validation_data=test_dataset.batch(2))

# Print the model's performance metrics
print(model.evaluate(test_dataset.batch(2)))