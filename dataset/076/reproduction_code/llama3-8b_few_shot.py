import tensorflow as tf
from modellib import MaskRCNN

# Define a custom layer that is not an instance of `tf.Variable`
class MyConv2D(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(MyConv2D, self).__init__()
        self.filters = filters

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(self.filters[0], (7, 7), activation='relu')
        self.bias1 = tf.Variable(1.0)  # Note: this is an instance of `tf.Variable`

    def call(self, x):
        return self.conv1(x) + self.bias1

# Create a model with the custom layer
model = MaskRCNN(mode="inference", model_dir='MODEL_DIR', config=config)
model.add(MyConv2D([64]))

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

print("Model loaded successfully")