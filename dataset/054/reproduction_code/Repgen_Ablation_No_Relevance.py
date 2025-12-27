import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow_decision_forests as tfdf

# Load data
train_data = pd.read_csv('path/to/train_data.csv')
test_data = pd.read_csv('path/to/test_data.csv')

# Ensure target column is of integer type
train_data['target'] = train_data['target'].astype(int)

# Check data types
print(train_data.dtypes)

# Preprocess input features (if needed)
# Assuming no preprocessing is required for this example

# Split train_data into training and validation sets with 80/20 ratio
train_df, val_df = train_test_split(train_data, test_size=0.2, random_state=42)

# Define a Gradient Boosted Trees model using TensorFlow Decision Forests library
gbt_model = tfdf.keras.GradientBoostedTreesModel(
    num_trees=100,
    max_depth=5,
    min_examples=10,
    subsample=0.8,
    validation_ratio=0.2,
    task=tfdf.keras.Task.CLASSIFICATION
)

# Compile the model with binary cross-entropy loss and Adam optimizer
gbt_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model on training data for 5 epochs with batch size of 32
history = gbt_model.fit(
    x=train_df.drop(columns=['target']),
    y=train_df['target'],
    validation_data=(val_df.drop(columns=['target']), val_df['target']),
    epochs=5,
    batch_size=32
)

# Evaluate the model on validation data and capture accuracy score
val_loss, val_accuracy = gbt_model.evaluate(val_df.drop(columns=['target']), val_df['target'])

# Print the final accuracy score to verify bug occurs
print(f'Validation Accuracy: {val_accuracy}')

# Assert that the accuracy score is less than expected threshold (e.g., 0.9)
assert val_accuracy < 0.9