from sklearn.model_selection import train_test_split

# Load your data (replace 'your_data' and 'labels' placeholders)
train_data, train_labels = load_data('pathology', 'train')
test_data, test_labels = load_data('pathology', 'test')

# Split into training and validation sets from the training data
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.1)

# Now train your model using only X_train for fitting but (incorrectly) use the test set for validation
model.fit(X_train, epochs=20, validation_data=test_data)  # Using test set as validation

# Evaluate the model on the completely unseen test data
loss = model.evaluate(test_data)