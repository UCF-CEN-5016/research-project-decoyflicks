import pandas as pd
import tensorflow as tf
import urllib.request  # Fixed undefined variable 'urllib'

BASE_PATH = "https://kdd.ics.uci.edu/databases/census-income/census-income"
CSV_HEADER = [
    l.decode("utf-8").split(":")[0].replace(" ", "_")
    for l in urllib.request.urlopen(f"{BASE_PATH}.names")
    if not l.startswith(b"|")
][2:]
CSV_HEADER.append("income_level")

train_data = pd.read_csv(f"{BASE_PATH}.data.gz", header=None, names=CSV_HEADER)
test_data = pd.read_csv(f"{BASE_PATH}.test.gz", header=None, names=CSV_HEADER)

TARGET_COLUMN_NAME = "income_level"
TARGET_LABELS = [" - 50000.", " 50000+."]

train_data[TARGET_COLUMN_NAME] = train_data[TARGET_COLUMN_NAME].map(TARGET_LABELS.index)
test_data[TARGET_COLUMN_NAME] = test_data[TARGET_COLUMN_NAME].map(TARGET_LABELS.index)

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# The following line reproduces the bug by using the deprecated append method
all_data = train_data.append(test_data)  # This will raise an AttributeError in pandas 2.0+