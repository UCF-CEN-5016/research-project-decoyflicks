import math
import urllib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_decision_forests as tfdf

BASE_PATH = "https://kdd.ics.uci.edu/databases/census-income/census-income"
CSV_HEADER = [
    l.decode("utf-8").split(":")[0].replace(" ", "_")
    for l in urllib.request.urlopen(f"{BASE_PATH}.names")
    if not l.startswith(b"|")
][2:]
CSV_HEADER.append("income_level")

train_data = pd.read_csv(
    f"{BASE_PATH}.data.gz",
    header=None,
    names=CSV_HEADER,
)
test_data = pd.read_csv(
    f"{BASE_PATH}.test.gz",
    header=None,
    names=CSV_HEADER,
)

# Define the missing variables to avoid undefined variable errors
TARGET_COLUMN_NAME = "income_level"  # Assuming this is the target column
TARGET_LABELS = ["<=50K", ">50K"]  # Example target labels
CATEGORICAL_FEATURE_NAMES = [col for col in CSV_HEADER if col != TARGET_COLUMN_NAME]  # Example categorical features

def prepare_dataframe(dataframe):
    dataframe[TARGET_COLUMN_NAME] = dataframe[TARGET_COLUMN_NAME].map(
        TARGET_LABELS.index
    )
    for feature_name in CATEGORICAL_FEATURE_NAMES:
        dataframe[feature_name] = dataframe[feature_name].astype(str)

prepare_dataframe(train_data)
prepare_dataframe(test_data)

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Refactor the append method to use pd.concat to avoid the AttributeError and FutureWarning
all_data = pd.concat([train_data, test_data], ignore_index=True)  # Preserve bug reproduction logic