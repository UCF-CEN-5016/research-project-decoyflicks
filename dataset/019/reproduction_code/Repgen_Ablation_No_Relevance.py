import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from official.modeling.networks import TNExpandCondense

# Set environment variable to suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from absl.testing import parameterized
except ImportError:
    parameterized = None

class TNExpandCondenseTest(tf.test.TestCase):
    def setUp(self):
        self.x_train, self.y_train = load_data()

    def _build_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(768,)),
            TNExpandCondense(),
            Dense(1, activation='sigmoid')
        ])
        return model

    @parameterized.parameters(*load_config('config.json')) if parameterized else None
    def test_train(self, config):
        model = self._build_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(self.x_train, self.y_train, epochs=5)
        self.assertLess(history.history['loss'][-1], 0.8)
        self.assertGreater(history.history['accuracy'][-1], 0.2)

def load_data():
    return tf.random.normal((100, 768)), tf.random.uniform((100, 1), maxval=2, dtype=tf.int32)

if __name__ == '__main__':
    if parameterized:
        tf.test.main()