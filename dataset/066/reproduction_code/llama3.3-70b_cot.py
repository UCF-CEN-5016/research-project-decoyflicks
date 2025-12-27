# Necessary imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set up minimal environment
def selective_kernel_feature_fusion(level1, level2, level3):
    # For demonstration purposes, this function simply returns the sum of the inputs
    return level1 + level2 + level3

# Define variables
level1_dau_2 = tf.random.normal([1, 256, 256, 3])  # Replace with actual values
level2_dau_2 = tf.random.normal([1, 128, 128, 3])  # Replace with actual values
level3_dau_2 = tf.random.normal([1, 64, 64, 3])  # Replace with actual values

# Triggering conditions
# Bug: Using level3_dau_2 instead of level2_dau_2
skff_buggy = selective_kernel_feature_fusion(level1_dau_2, level3_dau_2, level3_dau_2)

# Fix: Using level2_dau_2 instead of level3_dau_2
skff_fixed = selective_kernel_feature_fusion(level1_dau_2, level2_dau_2, level3_dau_2)

# Print the results
print("Buggy result:", skff_buggy)
print("Fixed result:", skff_fixed)