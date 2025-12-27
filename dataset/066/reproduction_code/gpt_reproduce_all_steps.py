import tensorflow as tf
from tensorflow import keras

def selective_kernel_feature_fusion(x1, x2, x3):
    def channel_attention(x, channels, reduction=8):
        avg_pool = keras.layers.GlobalAveragePooling2D()(x)
        dense1 = keras.layers.Dense(channels // reduction, activation="relu")(avg_pool)
        dense2 = keras.layers.Dense(channels, activation="sigmoid")(dense1)
        scale = keras.layers.Reshape((1, 1, channels))(dense2)
        return keras.layers.Multiply()([x, scale])

    channels = x1.shape[-1]
    x = keras.layers.Add()([x1, x2, x3])
    x = channel_attention(x, channels)
    return x

input_tensor = keras.Input(shape=(64, 64, 64))

level1_dau_2 = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(input_tensor)
level2_dau_2 = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(level1_dau_2)
level3_dau_2 = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(level2_dau_2)

# Buggy call (incorrect): level2_dau_2 is not used
output_buggy = selective_kernel_feature_fusion(level1_dau_2, level3_dau_2, level3_dau_2)

# Correct call: uses level2_dau_2 properly
output_fixed = selective_kernel_feature_fusion(level1_dau_2, level2_dau_2, level3_dau_2)

model_buggy = keras.Model(inputs=input_tensor, outputs=output_buggy)
model_fixed = keras.Model(inputs=input_tensor, outputs=output_fixed)

print("Buggy model summary:")
model_buggy.summary()

print("\nFixed model summary:")
model_fixed.summary()

import numpy as np
dummy_input = np.random.randn(1, 64, 64, 64).astype(np.float32)

output_buggy_val = model_buggy(dummy_input)
output_fixed_val = model_fixed(dummy_input)

print("\nOutput difference (sum abs):", tf.reduce_sum(tf.abs(output_fixed_val - output_buggy_val)).numpy())