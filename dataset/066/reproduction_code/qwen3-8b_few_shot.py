import tensorflow as tf

def selective_kernel_feature_fusion(a, b, c):
    # Dummy function to simulate the actual implementation
    return a + b + c

# Define the variables
level1_dau_2 = tf.random.normal([1, 1, 1, 1])
level2_dau_2 = tf.random.normal([1, 1, 1, 1])  # Unused variable
level3_dau_2 = tf.random.normal([1, 1, 1, 1])

# Function call with typo (using level3_dau_2 instead of level2_dau_2)
skff_ = selective_kernel_feature_fusion(level1_dau_2, level3_dau_2, level3_dau_2)

print("skff:", skff_)