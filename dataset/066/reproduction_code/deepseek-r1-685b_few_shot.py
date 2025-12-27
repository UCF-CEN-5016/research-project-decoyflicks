import tensorflow as tf
from tensorflow import keras

def selective_kernel_feature_fusion(*args):
    """Mock function to demonstrate the issue"""
    return sum(args)  # Simplified for demonstration

# Simplified MIRNet block to show the potential issue
def mirnet_block(inputs):
    # Mock DAU (Dual Attention Unit) outputs
    level1_dau_2 = keras.layers.Conv2D(32, 3)(inputs)
    level2_dau_2 = keras.layers.Conv2D(32, 3)(level1_dau_2)  # This variable is never used
    level3_dau_2 = keras.layers.Conv2D(32, 3)(level2_dau_2)
    
    # Potential typo - using level3_dau_2 twice instead of level2_dau_2
    skff = selective_kernel_feature_fusion(
        level1_dau_2, 
        level3_dau_2,  # Should this be level2_dau_2?
        level3_dau_2
    )
    return skff

# Test the block
inputs = keras.Input(shape=(256, 256, 3))
outputs = mirnet_block(inputs)
model = keras.Model(inputs, outputs)

print("Model built successfully, but potential typo exists")