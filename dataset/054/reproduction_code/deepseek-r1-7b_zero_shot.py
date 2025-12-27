# Assuming train_dataset contains only training data without any test split
model.fit(train_dataset, epochs=20)

val_dataset = ...  # Ensure val_dataset and test_dataset are different

model.fit(train_dataset, epochs=20, validation_data=val_dataset)

import tensorflow as tf
from tensorflow.keras import layers, Model

# Example data loading functions (replace with actual data)
def get_train_dataset():
    # Returns training dataset without test data
    pass

def get_val_dataset():
    # Returns separate validation dataset not involving test data
    pass

model.fit(get_train_dataset(), epochs=20, validation_data=get_val_dataset())