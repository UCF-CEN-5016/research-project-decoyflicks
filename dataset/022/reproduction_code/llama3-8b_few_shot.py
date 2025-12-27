import tensorflow as tf
from tensorflow.keras.applications import RandAugment

# Minimal setup to reproduce the issue
augmentor = RandAugment((2, 14))  # Define augmentation policy

print("Standard deviation:", augmentor.level_std)  # Expected non-zero value