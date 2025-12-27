import tensorflow as tf
from official.vision.beta.modeling.backbones import efficientnet
from official.vision.beta.modeling.heads import dense_prediction_heads

# Create simplified EfficientDet-like model
backbone = efficientnet.EfficientNet(model_id='efficientnet-b1')
input_tensor = tf.keras.layers.Input(shape=(640, 640, 3))
features = backbone(input_tensor)

# Add fpn and detection heads
fpn_features = dense_prediction_heads.FPNHead(min_level=3, max_level=7)(features)
class_outputs = dense_prediction_heads.RetinaClassHead(num_classes=90)(fpn_features)
box_outputs = dense_prediction_heads.RetinaBoxHead()(fpn_features)
model = tf.keras.Model(inputs=input_tensor, outputs=[class_outputs, box_outputs])

# Setup training
batch_size = 4
dummy_inputs = tf.random.normal((batch_size, 640, 640, 3))
dummy_boxes = tf.random.normal((batch_size, 100, 4))
dummy_classes = tf.random.uniform((batch_size, 100), maxval=90, dtype=tf.int32)

# Compile with problematic setup
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
loss = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=optimizer, loss=loss)

# Run training to trigger warning
model.fit(dummy_inputs, [dummy_classes, dummy_boxes], epochs=1, verbose=1)