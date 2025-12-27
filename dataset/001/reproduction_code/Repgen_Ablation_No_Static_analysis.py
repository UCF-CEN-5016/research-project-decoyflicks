import tensorflow as tf
from cluttered_mnist import transformer, spatial_transformer

batch_size = 100
height = 40
width = 40
channels = 1
num_classes = 10

# Load MNIST dataset and prepare it for training
(X_train, Y_train), (X_valid, Y_valid) = tf.keras.datasets.mnist.load_data()
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes)
Y_valid = tf.keras.utils.to_categorical(Y_valid, num_classes)

x = tf.placeholder(tf.float32, shape=[batch_size, height * width])
y = tf.placeholder(tf.float32, shape=[None, num_classes])

x_tensor = tf.reshape(x, [-1, height, width, channels])

# Two-layer localisation network
W_fc_loc1 = spatial_transformer.weight_variable([7*7*64, 20])
b_fc_loc1 = spatial_transformer.bias_variable([20])

h_conv2_flat = tf.reshape(h_conv2, [-1, 7*7*64])
h_fc_loc1 = tf.nn.tanh(tf.matmul(h_conv2_flat, W_fc_loc1) + b_fc_loc1)
h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)

W_fc_loc2 = spatial_transformer.weight_variable([20, 3 * 3])
b_fc_loc2 = spatial_transformer.bias_variable([3 * 3])

# Initial identity transformation
init_identity = tf.constant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=tf.float32)
initial_theta = tf.tile(tf.reshape(init_identity, [1, 6]), [batch_size, 1])

# Call transformer function
theta = h_fc_loc2 + initial_theta
out_size = (height, width)
output = transformer.transformer(x_tensor, theta, out_size)

# First convolutional layer
W_conv1 = spatial_transformer.weight_variable([5, 5, channels, 64])
b_conv1 = spatial_transformer.bias_variable([64])

h_conv1 = tf.nn.relu(spatial_transformer.conv2d(output, W_conv1) + b_conv1)

# Second convolutional layer
W_conv2 = spatial_transformer.weight_variable([5, 5, 64, 64])
b_conv2 = spatial_transformer.bias_variable([64])

h_conv2 = tf.nn.relu(spatial_transformer.conv2d(h_conv1, W_conv2) + b_conv2)

# Fully connected layer
W_fc1 = spatial_transformer.weight_variable([7 * 7 * 64, 1024])
b_fc1 = spatial_transformer.bias_variable([1024])

h_conv2_flat = tf.reshape(h_conv2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Softmax layer
W_fc2 = spatial_transformer.weight_variable([1024, num_classes])
b_fc2 = spatial_transformer.bias_variable([num_classes])

y_pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

# Gradient calculation
grads = tf.gradients(cross_entropy, tf.trainable_variables())

# Prediction and accuracy
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize all variables
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Training loop
iter_per_epoch = X_train.shape[0] // batch_size
for epoch in range(500):
    for i in range(iter_per_epoch):
        batch_xs = X_train[i * batch_size:(i + 1) * batch_size].reshape(batch_size, height * width)
        batch_ys = Y_train[i * batch_size:(i + 1) * batch_size]
        _, loss_value = sess.run([optimizer, cross_entropy], feed_dict={x: batch_xs, y: batch_ys})
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss_value}")
    
    acc = sess.run(accuracy, feed_dict={x: X_valid.reshape(-1, height * width), y: Y_valid})
    print(f"Validation Accuracy: {acc}")

# Verify 'nan' loss value
print("Final Loss Value:", loss_value)