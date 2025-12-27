import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
train_data = pd.read_csv('train_data.csv')

# Define constants
BATCH_SIZE = 256
NUM_TREES = 100
MAX_DEPTH = 30
MIN_EXAMPLES_PER_NODE = 10
SUBSAMPLE_RATIO = 0.8
VALIDATION_SPLIT = 0.2

# Preprocess function (target encoding)
def preprocess_data(data):
    # Placeholder for target encoding logic
    return data

train_data = preprocess_data(train_data)

# Split dataset into train and temporary set
temp_set, val_set = train_test_split(train_data, test_size=VALIDATION_SPLIT)

# Further split the temporary set into validation and test sets
val_set, test_set = train_test_split(temp_set, test_size=0.5)

# Create and compile model
gbt_model = tfdf.keras.GradientBoostedTreesModel(
    num_trees=NUM_TREES,
    max_depth=MAX_DEPTH,
    min_examples_per_node=MIN_EXAMPLES_PER_NODE,
    subsample=SUBSAMPLE_RATIO,
    task=tfdf.keras.Task.CLASSIFICATION
)
gbt_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Run experiment function
def run_experiment(model, train_data, val_data):
    history = model.fit(train_data, epochs=10, batch_size=BATCH_SIZE, validation_data=val_data)
    return history

history = run_experiment(gbt_model, train_set, val_set)

# Assert accuracy is less than 0.1
assert history.history['accuracy'][-1] < 0.1

# Monitor CPU memory usage
# Example command: top or htop