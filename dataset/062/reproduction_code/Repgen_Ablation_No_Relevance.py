import pandas as pd
import tensorflow as tf
from tensorflow import keras
import math
import tensorflow_decision_forests as tfdf

# Assuming necessary imports and datasets are already loaded and defined

def prepare_dataframe(dataframe):
    # Convert the target labels from string to integer.
    dataframe['TARGET_COLUMN_NAME'] = dataframe['TARGET_COLUMN_NAME'].map(
        TARGET_LABELS.index
    )
    # Cast the categorical features to string.
    for feature_name in CATEGORICAL_FEATURE_NAMES:
        dataframe[feature_name] = dataframe[feature_name].astype(str)

def run_experiment(model, train_data, test_data, num_epochs=1, batch_size=None):
    train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
        train_data, label='TARGET_COLUMN_NAME', weight='WEIGHT_COLUMN_NAME'
    )
    test_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
        test_data, label='TARGET_COLUMN_NAME', weight='WEIGHT_COLUMN_NAME'
    )

    model.fit(train_dataset, epochs=num_epochs, batch_size=batch_size)
    _, accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

def specify_feature_usages():
    feature_usages = []

    for feature_name in NUMERIC_FEATURE_NAMES:
        feature_usage = tfdf.keras.FeatureUsage(
            name=feature_name, semantic=tfdf.keras.FeatureSemantic.NUMERICAL
        )
        feature_usages.append(feature_usage)

    for feature_name in CATEGORICAL_FEATURE_NAMES:
        feature_usage = tfdf.keras.FeatureUsage(
            name=feature_name, semantic=tfdf.keras.FeatureSemantic.CATEGORICAL
        )
        feature_usages.append(feature_usage)

    return feature_usages

embedding_encoder = create_embedding_encoder(size=64)
run_experiment(
    create_nn_model(embedding_encoder), 
    train_data,
    test_data,
    num_epochs=5,
    batch_size=256
)