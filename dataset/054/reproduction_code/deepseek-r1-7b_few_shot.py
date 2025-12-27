import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Function to demonstrate the bug: Test set used as validation
def train_model(train_data, test_data, val_data, epochs=20):
    # Create a simple model architecture (dense layers)
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(10,)))
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train with different validation setups
    history_broken = model.fit(
        train_data,
        epochs=epochs,
        verbose=0,
        validation_data=test_data  # This is the bug: using test as val
    )
    
    return history_broken.history

# Generate sample data (training has more samples)
np.random.seed(42)
X_train = np.random.rand(1000, 10)  # Larger training set
X_val = np.random.rand(500, 10)
X_test = np.random.rand(300, 10)

# Train with bug and without bug scenarios
history_broken = train_model(X_train, X_test, X_val)
history_good = train_model(X_train, X_test, X_val)

print("With bug (test as val): Final training loss:", history_broken[-1])
print("Without bug (separate validation): Final training loss:", history_good[-1])