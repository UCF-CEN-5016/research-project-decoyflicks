import os
from keras import Input, Model, backend as K
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Add,
    Concatenate,
    Lambda,
    Activation,
    Dense,
    Reshape,
    Multiply,
    UpSampling2D,
)

def _conv_block(x, filters, kernel_size=3, activation="relu", name=None):
    x = Conv2D(filters, kernel_size, padding="same", activation=activation, name=name)(x)
    return x

def _residual_dau(x, filters, name_prefix="dau"):
    # A lightweight "Dual Attention Unit" like residual block
    residual = x
    x = _conv_block(x, filters, 3, activation="relu", name=f"{name_prefix}_conv1")
    x = _conv_block(x, filters, 3, activation=None, name=f"{name_prefix}_conv2")
    x = Add(name=f"{name_prefix}_add")([residual, x])
    x = Activation("relu", name=f"{name_prefix}_relu")(x)
    return x

def _downsample(x, filters, name_prefix="down"):
    x = MaxPooling2D(pool_size=(2, 2), name=f"{name_prefix}_pool")(x)
    x = _conv_block(x, filters, 3, activation="relu", name=f"{name_prefix}_conv")
    return x

def _upsample(x, filters, name_prefix="up"):
    x = UpSampling2D(size=(2, 2), name=f"{name_prefix}_upsample")(x)
    x = _conv_block(x, filters, 3, activation="relu", name=f"{name_prefix}_conv")
    return x

def selective_kernel_feature_fusion(f1, f2, f3, reduction=8, name_prefix="skff"):
    """
    Fuse three feature maps (same channels) using channel-wise attention across branches.
    Returns fused feature map with same number of channels as inputs.
    """
    # Assume f1, f2, f3 have same channel dimension
    channel_dim = int(K.int_shape(f1)[-1])

    # Sum features -> global context
    merged = Add(name=f"{name_prefix}_add_merged")([f1, f2, f3])

    # Global pooling
    context = GlobalAveragePooling2D(name=f"{name_prefix}_gap")(merged)  # (batch, channels)

    # Shared MLP to generate channel descriptors for each branch
    hidden_units = max(channel_dim // reduction, 1)
    shared_fc1 = Dense(hidden_units, activation="relu", name=f"{name_prefix}_fc1")
    shared_fc2 = Dense(channel_dim, activation=None, name=f"{name_prefix}_fc2")

    # For each branch get attention vector
    att1 = shared_fc2(shared_fc1(context))
    att2 = shared_fc2(shared_fc1(context))
    att3 = shared_fc2(shared_fc1(context))

    # Stack and softmax across branches (axis=1 after reshape)
    # Combine att vectors into shape (batch, branches, channels)
    def _stack_and_softmax(a1, a2, a3):
        stacked = K.stack([a1, a2, a3], axis=1)  # (batch, 3, channels)
        # softmax across branches
        weights = K.softmax(stacked, axis=1)
        return weights

    weights = Lambda(lambda tensors: _stack_and_softmax(tensors[0], tensors[1], tensors[2]),
                     name=f"{name_prefix}_softmax")([att1, att2, att3])

    # Split weights for each branch and apply
    def _extract_weights(weights_tensor, idx):
        # weights_tensor: (batch, 3, channels)
        return weights_tensor[:, idx, :]  # (batch, channels)

    w1 = Lambda(lambda w: _extract_weights(w, 0), name=f"{name_prefix}_w1")(weights)
    w2 = Lambda(lambda w: _extract_weights(w, 1), name=f"{name_prefix}_w2")(weights)
    w3 = Lambda(lambda w: _extract_weights(w, 2), name=f"{name_prefix}_w3")(weights)

    # reshape to (batch, 1, 1, channels) to multiply with feature maps
    w1 = Reshape((1, 1, channel_dim), name=f"{name_prefix}_reshape_w1")(w1)
    w2 = Reshape((1, 1, channel_dim), name=f"{name_prefix}_reshape_w2")(w2)
    w3 = Reshape((1, 1, channel_dim), name=f"{name_prefix}_reshape_w3")(w3)

    f1_weighted = Multiply(name=f"{name_prefix}_mul1")([f1, w1])
    f2_weighted = Multiply(name=f"{name_prefix}_mul2")([f2, w2])
    f3_weighted = Multiply(name=f"{name_prefix}_mul3")([f3, w3])

    fused = Add(name=f"{name_prefix}_fused")([f1_weighted, f2_weighted, f3_weighted])
    fused = Activation("relu", name=f"{name_prefix}_fused_relu")(fused)
    return fused

def MIRNet(input_shape=(None, None, 3)):
    """
    Construct a compact MIRNet-like model that preserves the original logic:
    - Multi-scale processing (level1, level2, level3)
    - Repeated DAU blocks per level
    - Selective Kernel Feature Fusion across levels
    """
    inputs = Input(shape=input_shape, name="input_image")

    # Initial conv
    init = _conv_block(inputs, 64, 3, activation="relu", name="init_conv")

    # Level 1 (full resolution)
    l1 = _residual_dau(init, 64, name_prefix="level1_dau_1")
    level1_dau_2 = _residual_dau(l1, 64, name_prefix="level1_dau_2")

    # Downsample to level 2
    l2_in = _downsample(level1_dau_2, 128, name_prefix="level2_down")
    l2 = _residual_dau(l2_in, 128, name_prefix="level2_dau_1")
    level2_dau_2 = _residual_dau(l2, 128, name_prefix="level2_dau_2")

    # Downsample to level 3
    l3_in = _downsample(level2_dau_2, 256, name_prefix="level3_down")
    l3 = _residual_dau(l3_in, 256, name_prefix="level3_dau_1")
    level3_dau_2 = _residual_dau(l3, 256, name_prefix="level3_dau_2")

    # Upsample level2 and level3 to match level1 resolution before fusion
    level2_up = _upsample(level2_dau_2, 64, name_prefix="level2_up_to_l1")
    level3_up = _upsample(_upsample(level3_dau_2, 128, name_prefix="level3_up_to_l2"), 64, name_prefix="level3_up_to_l1")

    # Ensure channels of all branches match (convert if needed)
    level1_proj = _conv_block(level1_dau_2, 64, 1, activation=None, name="proj_l1")
    level2_proj = _conv_block(level2_up, 64, 1, activation=None, name="proj_l2")
    level3_proj = _conv_block(level3_up, 64, 1, activation=None, name="proj_l3")

    # Selective Kernel Feature Fusion (SKFF)
    out7 = selective_kernel_feature_fusion(level1_proj, level2_proj, level3_proj, reduction=8, name_prefix="skff")

    # Final output conv to restore 3-channel image
    final = Conv2D(3, kernel_size=3, padding="same", activation="sigmoid", name="final_conv")(out7)

    model = Model(inputs=inputs, outputs=final, name="MIRNet_like")
    return model

if __name__ == "__main__":
    input_shape = (256, 256, 3)  # Adjust based on your input size
    model = MIRNet(input_shape)
    model.summary()