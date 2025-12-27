import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def selective_kernel_feature_fusion(a, b, c):
    return a + b + c

def build_model():
    inputs = keras.Input(shape=(None, None, 3))
    level1_dau_2 = layers.Conv2D(32, 3, padding="same")(inputs)
    level2_dau_2 = layers.Conv2D(32, 3, padding="same")(level1_dau_2)
    level3_dau_2 = layers.Conv2D(32, 3, padding="same")(level2_dau_2)
    skff_ = selective_kernel_feature_fusion(level1_dau_2, level3_dau_2, level3_dau_2)
    outputs = layers.Conv2D(3, 3, padding="same")(skff_)
    return keras.Model(inputs, outputs)

model = build_model()
model.summary()