import numpy as np
from keras.losses import sparse_categorical_crossentropy
from keras.models import Sequential
from keras.layers import Dense

# Define the model
model = Sequential()
model.add(Dense(10, activation='softmax'))

# Define the labels
labels = np.array([[1, 0, 0], [0, 1, 0]])

# Compute the loss
loss = sparse_categorical_crossentropy(labels, labels)

print(loss)