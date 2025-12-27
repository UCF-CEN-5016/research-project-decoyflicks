import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def selective_kernel_feature_fusion(f1, f2, f3):
    """Mock fusion function for reproduction"""
    print(f"Fusing features with shapes: {f1.shape}, {f2.shape}, {f3.shape}")
    return (f1 + f2 + f3) / 3

# Mock DAU outputs with different shapes to make the bug obvious
level1_dau_2 = tf.random.normal((1, 64, 64, 32))  # Higher resolution
level2_dau_2 = tf.random.normal((1, 32, 32, 64))  # Mid resolution
level3_dau_2 = tf.random.normal((1, 16, 16, 128))  # Lower resolution

# Original buggy version - using level3_dau_2 twice
print("Buggy version:")
skff_buggy = selective_kernel_feature_fusion(level1_dau_2, level3_dau_2, level3_dau_2)

# Fixed version - using all three levels correctly
print("\nFixed version:")
skff_fixed = selective_kernel_feature_fusion(level1_dau_2, level2_dau_2, level3_dau_2)