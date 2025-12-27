import tensorflow as tf
from object_detection import DetectionModel

tf.random.set_seed(42)

batch_size = 2
height, width = 10, 10

# Create random uniform input data
input_data = tf.random.uniform((batch_size, height, width, 3), maxval=256, dtype=tf.int32)
true_image_shapes = tf.constant([[height, width] for _ in range(batch_size)], dtype=tf.float32)

# Ground truth boxes and classes for prediction
groundtruth_boxes = tf.constant([[[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]], [[5.0, 5.0, 6.0, 6.0]]], dtype=tf.float32)
groundtruth_classes = tf.constant([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]], dtype=tf.float32)

# Set up the DetectionModel
model_config = {
    "faster_rcnn": {
        "num_classes": 37
    }
}
configs = {"model": model_config, "train_config": {}, "train_input_config": {}}
detection_model = DetectionModel(configs["model"])

# Provide input data and ground truth information to the model's predict method
predictions = detection_model.predict(input_data, true_image_shapes, groundtruth_boxes, groundtruth_classes)

# Calculate losses using the model's loss method with random example sampling set to True
losses = detection_model.loss(predictions, groundtruth_boxes, groundtruth_classes, random_example_sampling=True)

# Assert that localization_loss is close to 0.0
tf.assert_near(losses["localization_loss"], 0.0)

# Assert that classification_loss is close to (batch_size * number_of_anchors * (num_classes + 1) * ln(2))
classification_loss_expected = batch_size * 3 * (model_config["faster_rcnn"]["num_classes"] + 1) * tf.math.log(2)
tf.assert_near(losses["classification_loss"], classification_loss_expected)

# Monitor the memory usage during execution
with tf.profiler.experimental.Profile("/tmp/logdir"):
    sess = tf.compat.v1.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    predictions_val, losses_val = sess.run([predictions, losses])

# Verify that GPU memory usage exceeds expected threshold for a batch of size 2
print(sess.run(tf.config.experimental.get_memory_info("GPU")))