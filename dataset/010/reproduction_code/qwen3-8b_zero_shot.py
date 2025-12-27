import tensorflow as tf
import numpy as np

# 1. Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid')  # Use sigmoid for binary classification
])
model.compile(optimizer='adam', loss='binary_crossentropy')

# 2. Train the model
# Training data: 100 samples, 10 features
x_train = np.random.normal(0, 1, (100, 10))
y_train = np.random.choice([0, 1], size=(100, 1))  # Binary labels

# Train the model for one epoch
model.fit(x_train, y_train, epochs=1, verbose=0)

# 3. Create evaluation data where predictions are exactly equal to the true labels
x_eval = np.random.normal(0, 1, (1, 10))  # One sample for evaluation
y_eval = model.predict(x_eval).round()  # Round to get binary predictions

# 4. Evaluate the model
loss = model.evaluate(x_eval, y_eval, verbose=0)
print("Loss:", loss)