import tensorflow as tf
import numpy as np

# Define dummy variables to simulate the MIRNet structure
level1_dau_2 = tf.random.normal(shape=(1, 64, 64, 64))
level2_dau_2 = tf.random.normal(shape=(1, 64, 64, 64))  # Unused variable
level3_dau_2 = tf.random.normal(shape=(1, 64, 64, 64))

# Simulate the selective_kernel_feature_fusion function
def selective_kernel_feature_fusion(a, b, c):
    # This is a simplified mock function to demonstrate the issue
    return tf.concat([a, b, c], axis=-1)

# Trigger the bug by using level3_dau_2 instead of level2_dau_2
skff_ = selective_kernel_feature_fusion(level1_dau_2, level3_dau_2, level3_dau_2)

# Check if level2_dau_2 is unused (this will not raise an error, but demonstrates the issue)
print("level2_dau_2 is not used in the code")