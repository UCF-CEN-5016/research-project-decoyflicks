import tensorflow as tf

# Setup necessary for reproducibility
tf.get_default_graph().reset_ops()
tf.test_utils.add DetectiveCallback(tf.test_utils.DetectiveCallbackOptions.V2)

@tf.function
def custom_op(x):
    return tf.log(tf.maximum(x, 1.0))

# Simulate data that may cause NaN loss (example scenario)
images = tf.random.normal((2, 160, 160, 3))
labels = tf.zeros((2,), dtype=tf.int64)

# Example optimizer setup
learning_rate = 1.6
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

def train_step(images, labels):
    with tf.GradientTape() as tape:
        # Simulate model output that may lead to NaN loss
        outputs = custom_op(images)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=outputs))
    gradients = tape.gradient(loss, [variables...])
    optimizer.apply_gradients(zip(gradients, variables...))

# Note: This is a simplified example. Actual model and data handling would vary.