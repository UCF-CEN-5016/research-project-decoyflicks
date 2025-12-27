import tensorflow as tf

def selective_kernel_feature_fusion(level1, level2, level3):
    return tf.concat([level1, level2, level3], axis=-1)

def main():
    level1_dau_2 = tf.random.normal([1, 256, 256, 64])
    level2_dau_2 = tf.random.normal([1, 128, 128, 128])
    level3_dau_2 = tf.random.normal([1, 64, 64, 256])

    skff_ = selective_kernel_feature_fusion(level1_dau_2, level3_dau_2, level3_dau_2)
    print(skff_.shape)

    skff_corrected = selective_kernel_feature_fusion(level1_dau_2, level2_dau_2, level3_dau_2)
    print(skff_corrected.shape)

if __name__ == "__main__":
    main()