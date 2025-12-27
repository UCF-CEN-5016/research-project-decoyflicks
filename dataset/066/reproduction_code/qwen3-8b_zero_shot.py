

import tensorflow as tf

def selective_kernel_feature_fusion(a, b, c):
    return tf.concat([a, b, c], axis=-1)

level1_dau_2 = tf.constant(0.0)
level2_dau_2 = tf.constant(0.0)
level3_dau_2 = tf.constant(0.0)

skff_ = selective_kernel_feature_fusion(level1_dau_2, level3_dau_2, level3_dau_2)