import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split data into training and validation sets (WRONG!)
train_dataset = (X_train, y_train)
test_dataset = (X_test, y_test)

# Train model with wrong validation set
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(train_dataset, epochs=20, validation_data=test_dataset)

print("Model performance:", model.evaluate(test_dataset)[1])  # Should be low