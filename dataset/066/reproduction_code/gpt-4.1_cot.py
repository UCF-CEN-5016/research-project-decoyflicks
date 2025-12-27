import tensorflow as tf
from tensorflow.keras import layers, Model

# Dummy selective_kernel_feature_fusion function to mimic expected behavior
def selective_kernel_feature_fusion(x1, x2, x3):
    # Just concatenate for demonstration
    return layers.Concatenate()([x1, x2, x3])

# Dummy DAU block to simulate the outputs (level1_dau_2, level2_dau_2, level3_dau_2)
def dummy_dau_block(name):
    # Return a tensor with shape (batch, height, width, channels)
    input_tensor = layers.Input(shape=(32, 32, 16), name=f"{name}_input")
    x = layers.Conv2D(16, 3, padding='same', activation='relu', name=f"{name}_conv")(input_tensor)
    return Model(input_tensor, x, name=name)

# Create dummy inputs representing the outputs of the DAU blocks
level1_dau_2_model = dummy_dau_block("level1_dau_2")
level2_dau_2_model = dummy_dau_block("level2_dau_2")
level3_dau_2_model = dummy_dau_block("level3_dau_2")

# Create input tensors
input1 = tf.random.normal((1, 32, 32, 16))
input2 = tf.random.normal((1, 32, 32, 16))
input3 = tf.random.normal((1, 32, 32, 16))

# Get the outputs from these dummy DAU blocks
level1_dau_2 = level1_dau_2_model(input1)
level2_dau_2 = level2_dau_2_model(input2)
level3_dau_2 = level3_dau_2_model(input3)

# Buggy code snippet:
# skff_ = selective_kernel_feature_fusion(level1_dau_2, level3_dau_2, level3_dau_2)

# Suspected correct code snippet:
skff_correct = selective_kernel_feature_fusion(level1_dau_2, level2_dau_2, level3_dau_2)

print("Shapes with buggy fusion:",
      selective_kernel_feature_fusion(level1_dau_2, level3_dau_2, level3_dau_2).shape)
print("Shapes with corrected fusion:",
      skff_correct.shape)