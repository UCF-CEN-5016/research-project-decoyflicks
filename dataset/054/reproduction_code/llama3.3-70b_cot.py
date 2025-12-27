# Import necessary libraries
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Set up minimal environment
# Create a dummy dataset
np.random.seed(0)
X = np.random.rand(100, 10)  # 100 samples, 10 features
y = np.random.randint(0, 2, 100)  # Binary classification

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a simple neural network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Triggering conditions: using the test set as the validation set
# This is where the bug is introduced
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# To fix the bug, we should split the data into training, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Now, use the validation set for validation
model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))