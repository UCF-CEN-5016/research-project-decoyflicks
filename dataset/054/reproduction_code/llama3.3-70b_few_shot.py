import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Sample dataset
X = np.random.rand(100, 10)
y = np.random.rand(100)

# Split into training and test sets
train_dataset, test_dataset, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a simple model
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Incorrectly use the test set as the validation set
model.fit(train_dataset, train_labels, epochs=20, validation_data=(test_dataset, test_labels))

# Print the model's performance on the "validation" set
print("Model performance on 'validation' set:")
print(model.evaluate(test_dataset, test_labels))