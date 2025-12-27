import tensorflow as tf
import numpy as np

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='sigmoid')  # Use sigmoid for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def generate_data(samples, features):
    x_data = np.random.normal(0, 1, (samples, features))
    y_data = np.random.choice([0, 1], size=(samples, 1))  # Binary labels
    return x_data, y_data

def train_model(model, x_train, y_train, epochs):
    model.fit(x_train, y_train, epochs=epochs, verbose=0)

def evaluate_model(model, x_eval, y_eval):
    loss = model.evaluate(x_eval, y_eval, verbose=0)
    print("Loss:", loss)

# 1. Define and create the model
model = create_model()

# 2. Train the model
x_train, y_train = generate_data(100, 10)
train_model(model, x_train, y_train, epochs=1)

# 3. Create evaluation data where predictions are exactly equal to the true labels
x_eval = np.random.normal(0, 1, (1, 10))  # One sample for evaluation
y_eval = model.predict(x_eval).round()  # Round to get binary predictions

# 4. Evaluate the model
evaluate_model(model, x_eval, y_eval)