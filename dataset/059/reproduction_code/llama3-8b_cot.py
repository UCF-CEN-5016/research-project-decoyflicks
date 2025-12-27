import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Create a simple dataset (e.g., MNIST)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Reshape data to fit Keras models
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

# Normalize pixel values (optional)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Define a simple Keras model with some basic components
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

if __name__ == '__main__':
    # Try to fit the model (this should trigger the error)
    model.fit(X_train, y_train, epochs=1)

    print("Model fitting result:", model.evaluate(X_test, y_test))