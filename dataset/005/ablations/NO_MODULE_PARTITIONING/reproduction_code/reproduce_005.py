# Requirements: tf-models-official

import tensorflow as tf
from tf_keras import applications

def test_mobilenet():
    try:
        model = applications.MobileNetV2(weights='imagenet', include_top=True)
        input_data = tf.random.uniform((1, 224, 224, 3))
        output = model(input_data)
    except ImportError as e:
        print(e)

if __name__ == "__main__":
    test_mobilenet()