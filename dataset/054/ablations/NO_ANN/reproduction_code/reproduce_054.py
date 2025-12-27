import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_decision_forests as tfdf

# Define the base path for the dataset
BASE_PATH = 'https://kdd.ics.uci.edu/databases/census-income/census-income'

# Load the training and test datasets
train_data = pd.read_csv(f'{BASE_PATH}.data.gz', header=None)
test_data = pd.read_csv(f'{BASE_PATH}.test.gz', header=None)

# Define target column and labels
TARGET_COLUMN_NAME = 'income_level'
TARGET_LABELS = [' - 50000.', ' 50000+.']
train_data[TARGET_COLUMN_NAME] = train_data[TARGET_COLUMN_NAME].map(TARGET_LABELS.index)
test_data[TARGET_COLUMN_NAME] = test_data[TARGET_COLUMN_NAME].map(TARGET_LABELS.index)

# Define feature names (these should be defined based on the dataset)
CATEGORICAL_FEATURE_NAMES = []  # Placeholder: Define your categorical feature names here
NUMERIC_FEATURE_NAMES = []       # Placeholder: Define your numeric feature names here

# Convert categorical features to string type
for feature_name in CATEGORICAL_FEATURE_NAMES:
    train_data[feature_name] = train_data[feature_name].astype(str)
    test_data[feature_name] = test_data[feature_name].astype(str)

# Model parameters
NUM_TREES = 250
MAX_DEPTH = 5
MIN_EXAMPLES = 6
SUBSAMPLE = 0.65
VALIDATION_RATIO = 0.1

def specify_feature_usages():
    feature_usages = []
    for feature_name in NUMERIC_FEATURE_NAMES:
        feature_usage = tfdf.keras.FeatureUsage(name=feature_name, semantic=tfdf.keras.FeatureSemantic.NUMERICAL)
        feature_usages.append(feature_usage)
    for feature_name in CATEGORICAL_FEATURE_NAMES:
        feature_usage = tfdf.keras.FeatureUsage(name=feature_name, semantic=tfdf.keras.FeatureSemantic.CATEGORICAL)
        feature_usages.append(feature_usage)
    return feature_usages

# Initialize the Gradient Boosted Trees model
gbt_model = tfdf.keras.GradientBoostedTreesModel(
    features=specify_feature_usages(),
    num_trees=NUM_TREES,
    max_depth=MAX_DEPTH,
    min_examples=MIN_EXAMPLES,
    subsample=SUBSAMPLE,
    validation_ratio=VALIDATION_RATIO,
    task=tfdf.keras.Task.CLASSIFICATION
)

# Compile the model with accuracy metric
gbt_model.compile(metrics=[keras.metrics.BinaryAccuracy(name='accuracy')])

def run_experiment(model, train_data, test_data, num_epochs=1, batch_size=None):
    # Convert pandas DataFrame to TensorFlow dataset
    train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(train_data, label=TARGET_COLUMN_NAME)
    test_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(test_data, label=TARGET_COLUMN_NAME)
    
    # Train the model using the test dataset as validation data (bug reproduction)
    model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
    
    # Evaluate the model on the test dataset
    _, accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

# Run the experiment with the model
run_experiment(gbt_model, train_data, test_data)