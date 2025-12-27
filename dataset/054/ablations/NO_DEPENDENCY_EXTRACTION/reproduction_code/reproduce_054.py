import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_decision_forests as tfdf

BASE_PATH = 'https://kdd.ics.uci.edu/databases/census-income/census-income'
train_data = pd.read_csv(f'{BASE_PATH}.data.gz', header=None)
test_data = pd.read_csv(f'{BASE_PATH}.test.gz', header=None)

TARGET_COLUMN_NAME = 'income_level'
TARGET_LABELS = [' - 50000.', ' 50000+.']
train_data[TARGET_COLUMN_NAME] = train_data[TARGET_COLUMN_NAME].map(TARGET_LABELS.index)
test_data[TARGET_COLUMN_NAME] = test_data[TARGET_COLUMN_NAME].map(TARGET_LABELS.index)

NUM_TREES = 250
MAX_DEPTH = 5
MIN_EXAMPLES = 6
SUBSAMPLE = 0.65
VALIDATION_RATIO = 0.1

# Define feature names (these should be defined to avoid undefined variable errors)
NUMERIC_FEATURE_NAMES = ['feature1', 'feature2', 'feature3']  # Replace with actual numeric feature names
CATEGORICAL_FEATURE_NAMES = ['feature4', 'feature5']  # Replace with actual categorical feature names

def specify_feature_usages():
    feature_usages = []
    for feature_name in NUMERIC_FEATURE_NAMES:
        feature_usage = tfdf.keras.FeatureUsage(name=feature_name, semantic=tfdf.keras.FeatureSemantic.NUMERICAL)
        feature_usages.append(feature_usage)
    for feature_name in CATEGORICAL_FEATURE_NAMES:
        feature_usage = tfdf.keras.FeatureUsage(name=feature_name, semantic=tfdf.keras.FeatureSemantic.CATEGORICAL)
        feature_usages.append(feature_usage)
    return feature_usages

gbt_model = tfdf.keras.GradientBoostedTreesModel(
    features=specify_feature_usages(),
    num_trees=NUM_TREES,
    max_depth=MAX_DEPTH,
    min_examples=MIN_EXAMPLES,
    subsample=SUBSAMPLE,
    validation_ratio=VALIDATION_RATIO,
    task=tfdf.keras.Task.CLASSIFICATION
)

gbt_model.compile(metrics=[keras.metrics.BinaryAccuracy(name='accuracy')])
train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(train_data, label=TARGET_COLUMN_NAME)
test_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(test_data, label=TARGET_COLUMN_NAME)

# Bug reproduction: using test dataset as validation dataset
gbt_model.fit(train_dataset, epochs=20, validation_data=test_dataset)  # This line reproduces the bug
_, accuracy = gbt_model.evaluate(test_dataset, verbose=0)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")