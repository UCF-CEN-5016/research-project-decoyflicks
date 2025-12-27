import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from autoencoder.Utils import Autoencoder  # Fixed undefined variable issue

np.random.seed(42)

n_input = 784
n_hidden = 256
batch_size = 128
epochs = 50

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32) / 255.0
X_train = X_train.reshape(-1, n_input)
X_test = X_test.astype(np.float32) / 255.0
X_test = X_test.reshape(-1, n_input)

# Split the training data into training and validation sets
X_train, X_val = train_test_split(X_train, test_size=0.2)

# Initialize the Autoencoder model
autoencoder = Autoencoder(n_input=n_input, n_hidden=n_hidden, optimizer=tf.train.AdamOptimizer())

# Training loop
for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        batch = X_train[i:i + batch_size]
        loss = autoencoder.partial_fit(batch)
        print(f'Epoch: {epoch}, Batch: {i // batch_size}, Loss: {loss}')
        
        # Check for NaN loss values
        if np.isnan(loss):
            print("NaN loss detected!")
            break

# Calculate and print the final loss on the validation set
print("Final loss after training:", autoencoder.calc_total_cost(X_val))