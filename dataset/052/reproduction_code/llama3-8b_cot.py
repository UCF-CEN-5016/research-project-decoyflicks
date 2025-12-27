import tensorflow as tf
from tensorflow import keras
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create sample dataset for training
train_dataset = ...

# Define model architecture and compile it with optimizer and loss function
model = keras.Sequential([...])  # Define your model architecture here
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Train the model using fit method
model.fit(train_dataset, epochs=10)

# Sample inference using the trained model (this is where the error occurs)
sample_input = ...  # tokenize and convert to ids
output = model.predict(sample_input)
prediction = np.argmax(output, axis=-1)[0]
prediction = [mapping[i] for i in prediction]

print(prediction)  # This should print an array of predicted labels