import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set TensorFlow version
tf.__version__ = "2.11.0"

# Parameters
batch_size = 32
epochs = 10

# Sample training dataset
train_data = np.random.rand(1000, 4).astype(np.float32)
train_labels = np.random.randint(0, 3, size=(1000,))  # Assuming 3 classes

# Define a simple Keras model for NER
ner_model = keras.Sequential([
    layers.Input(shape=(None, 4)),
    layers.Dense(3, activation='softmax')  # 3 classes for softmax
])

# Compile the model
ner_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Create TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(batch_size)

# Fit the model
ner_model.fit(train_dataset, epochs=epochs)

# Tokenization function
def tokenize_and_convert_to_ids(text):
    return text.lower().split()

# Sample input string
sample_input_string = "eu rejects german call to boycott british lamb"

# Tokenize the input string
tokenized_input = tokenize_and_convert_to_ids(sample_input_string)

# Reshape the tokenized input
sample_input = tf.reshape(tf.convert_to_tensor(tokenized_input), shape=[1, -1])

# Predict
predictions = ner_model.predict(sample_input)

# Get predicted class indices
predicted_indices = np.argmax(predictions, axis=-1)

# Print predicted labels
print(predicted_indices)