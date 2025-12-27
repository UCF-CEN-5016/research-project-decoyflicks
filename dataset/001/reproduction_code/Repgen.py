import tensorflow as tf
from spatial_transformer import transformer
from tf_utils import bias_variable, dense_to_one_hot, weight_variable
import numpy as np  # Import numpy at the top of the file

# Set random seed for reproducibility
tf.random.set_seed(42)

# Load dataset
dataset = np.load('./data/mnist_sequence1_sample_5distortions5x5.npz')
X_train, y_train, X_valid, y_valid, X_test, y_test = dataset['X_train'], dataset['y_train'], dataset['X_valid'], dataset['y_valid'], dataset['X_test'], dataset['y_test']

# Convert labels to one-hot encoding
y_train = dense_to_one_hot(y_train, n_classes=10)
y_valid = dense_to_one_hot(y_valid, n_classes=10)
y_test = dense_to_one_hot(y_test, n_classes=10)

# Define batch size and placeholders
batch_size = 32
x = tf.placeholder(tf.float32, shape=[None, 1600])
y = tf.placeholder(tf.float32, shape=[None, 10])

# Reshape input data
x_reshaped = tf.reshape(x, [-1, 40, 40, 1])

# Initialize weight and bias functions
W_fc_loc1, b_fc_loc1 = weight_variable([7 * 7 * 32, 20]), bias_variable([20])
W_fc_loc2, b_fc_loc2 = weight_variable([20, 6], name='W_fc_loc2')

# Set initial values for b_fc_loc2 to identity transformation matrix flattened
b_fc_loc2.initializer.run(session=tf.compat.v1.Session())

# Define two-layer localisation network
h_fc_loc1 = tf.nn.tanh(tf.matmul(tf.reshape(x_reshaped, [-1, 7 * 7 * 32]), W_fc_loc1) + b_fc_loc1)
h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1, W_fc_loc2) + b_fc_loc2)

# Apply spatial transformation
theta = tf.reshape(h_fc_loc2, [-1, 2, 3])
x_trans = transformer(x_reshaped, theta, out_size=[40, 40])

# Define convolutional layers
W_conv1, b_conv1 = weight_variable([5, 5, 1, 32]), bias_variable([32])
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_trans, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
W_conv2, b_conv2 = weight_variable([5, 5, 32, 64]), bias_variable([64])
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2)

# Flatten output for fully connected layers
h_conv2_flat = tf.reshape(h_conv2, [-1, 8 * 8 * 64])

# Define fully connected layers
W_fc1, b_fc1 = weight_variable([8 * 8 * 64, 1024]), bias_variable([1024])
h_fc1 = tf.nn.tanh(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, rate=1 - keep_prob)

# Output fully connected layer for classification
W_fc_out, b_fc_out = weight_variable([1024, 10]), bias_variable([10])
y_pred = tf.matmul(h_fc1_drop, W_fc_out) + b_fc_out

# Compute cross-entropy loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_pred))

# Initialize variables and start TensorFlow session
init_op = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init_op)
    n_epochs = 500
    iter_per_epoch = int(X_train.shape[0] / batch_size)
    indices = np.arange(X_train.shape[0])
    
    for epoch in range(n_epochs):
        np.random.shuffle(indices)
        for i in range(iter_per_epoch - 1):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch_xs, batch_ys = X_train[indices[start:end]], y_train[indices[start:end]]
            if i % 10 == 0:
                test_loss = sess.run(cross_entropy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
                print(f"Epoch {epoch + 1}, Iteration {i}, Test Loss: {test_loss}")
            sess.run([W_fc_loc2.assign(tf.identity_transform.flatten()), W_fc_loc1.assign(tf.random_normal([7 * 7 * 32, 20]))], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.8})
        valid_loss = sess.run(cross_entropy, feed_dict={x: X_valid, y: y_valid, keep_prob: 1.0})
        print(f"Epoch {epoch + 1}, Validation Loss: {valid_loss}")