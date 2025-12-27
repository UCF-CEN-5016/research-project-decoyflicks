import tensorflow as tf
from spatial_transformer import transformer
from tf_utils import bias_variable, dense_to_one_hot, weight_variable

# Load MNIST data
(X_train, y_train), (X_valid, y_valid) = tf.keras.datasets.mnist.load_data()
X_test, y_test = X_valid, y_valid  # Reusing validation for simplicity

# Reshape and normalize data
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_valid = X_valid.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# One-hot encode labels
y_train = dense_to_one_hot(y_train)
y_valid = dense_to_one_hot(y_valid)

# Define placeholders and parameters
batch_size = 32
n_epochs = 50
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Reshape input tensor
x_tensor = tf.reshape(x, [-1, 28, 28, 1])

# Localization network
W_fc_loc1 = weight_variable([784, 20])
b_fc_loc1 = bias_variable([20])
h_fc_loc1 = tf.nn.tanh(tf.matmul(tf.reshape(x, [-1, 784]), W_fc_loc1) + b_fc_loc1)
keep_prob = tf.placeholder(tf.float32)
h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
W_fc_loc2 = weight_variable([20, 36])
b_fc_loc2 = bias_variable([36])

# Spatial transformer network
init = tf.zeros([batch_size, 6], dtype=tf.float32)  # Initialize theta to identity matrix
h_trans = transformer(x_tensor, init)

# Convolutional layers
W_conv1 = weight_variable([3, 3, 1, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(tf.nn.conv2d(h_trans, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

W_conv2 = weight_variable([3, 3, 16, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

# Fully connected layers
h_conv2_flat = tf.reshape(h_conv2, [-1, 10 * 10 * 16])
W_fc1 = weight_variable([10 * 10 * 16, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Loss function and optimizer
cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy, var_list=[b_fc_loc2])

# Initialize variables
init_op = tf.global_variables_initializer()

# Training loop
iter_per_epoch = 100
train_size = X_train.shape[0]
with tf.Session() as sess:
    sess.run(init_op)
    for epoch_i in range(n_epochs):
        for iter_i in range(iter_per_epoch - 1):
            batch_xs, batch_ys = X_train[iter_i * batch_size:(iter_i + 1) * batch_size], y_train[iter_i * batch_size:(iter_i + 1) * batch_size]
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
        if iter_i % 10 == 0:
            print('Epoch:', epoch_i, 'Loss:', cross_entropy.eval(feed_dict={x: batch_xs, y: batch_ys}))
    valid_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_valid, 1)), tf.float32))
    print('Validation Accuracy:', sess.run(valid_accuracy, feed_dict={x: X_valid, y: y_valid, keep_prob: 1.0}))

# Monitor GPU memory usage
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print('GPU Memory Usage:', info.used // (1024 ** 3), 'GB')