import tensorflow as tf
from tensorflow.keras import layers

def selective_kernel_feature_fusion(x1, x2, x3):
    return x1 + x2 + x3

level1_dau_2 = tf.constant(1.0)
level2_dau_2 = tf.constant(2.0)
level3_dau_2 = tf.constant(3.0)

# Original code (potentially buggy)
skff_buggy = selective_kernel_feature_fusion(level1_dau_2, level3_dau_2, level3_dau_2)

# Suggested fix
skff_fixed = selective_kernel_feature_fusion(level1_dau_2, level2_dau_2, level3_dau_2)

print("Buggy output:", skff_buggy.numpy())
print("Fixed output:", skff_fixed.numpy())