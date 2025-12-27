import tensorflow as tf
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, \
    concatenate, Conv2D, AveragePooling2D, GlobalAveragePooling2D, \
    BatchNormalization, LeakyReLU, SeparableConv2D, UpSampling2D, \
    DepthwiseConv2D

def MIRNet(input_shape=(None, None, 3)):
    inputs = Input(shape=input_shape)
    
    # Pretrained Models
    vgg16_model = tf.keras.applications.VGG16(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet'
    )
    mnet_base_model = tf.keras.applications.MOBILENETV2(
        include_top=False,
        input_shape=(None, None, 3),
        weights='imagenet',
        input_tensor=inputs
    )
    
    # Feature extraction
    level1_dau_1 = vgg16_model.output
    level1_dau_2 = tf.keras.layers.GlobalAveragePooling2D()(level1_dau_1)
    ...
    level2_dau_2 = ...  # Ensure this layer is properly created
    
    # ... (other layers and features)

    skff_ = selective_kernel_feature_fusion(level1_dau_2, level2_dau_2, level3_dau_2)  # Fixed both instances