import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

def load_model():
    model = MobileNetV2(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
    return model

def test_model():
    model = load_model()
    input_data = tf.random.uniform((1, 224, 224, 3), minval=0.0, maxval=1.0)
    output = model.predict(input_data)
    print(output.shape)

if __name__ == '__main__':
    test_model()