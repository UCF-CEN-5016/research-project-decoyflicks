import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Assuming you have your dataset loaded into `data` and labels into `labels`
# For example, data could be a list of images and labels their corresponding classes

# Split the dataset into training and validation datasets
train_data, validation_data = train_test_split(data, labels, test_size=0.2)

# Define the model architecture (using TensorFlow Keras)
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_data, labels, epochs=10, batch_size=32, validation_data=(validation_data, labels))

# Evaluate the model on test data (assuming separate test_data and test_labels are available)
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {test_accuracy}")