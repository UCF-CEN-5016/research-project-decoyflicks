import numpy as np
import tensorflow as tf
from cluttered_mnist import transformer, dense_to_one_hot

batch_size = 100
img_rows, img_cols, channels = 40, 40, 1

data_path = './data/mnist_sequence1_sample_5distortions5x5.npz'
npzfile = np.load(data_path)
X_train = npzfile['X_train']
y_train = npzfile['y_train']
X_valid = npzfile['X_valid']
y_valid = npzfile['y_valid']
X_test = npzfile['X_test']
y_test = npzfile['y_test']

y_train = dense_to_one_hot(y_train, n_classes=10)
y_valid = dense_to_one_hot(y_valid, n_classes=10)
y_test = dense_to_one_hot(y_test, n_classes=10)

x = tf.placeholder(tf.float32, [None, 1600])
y = tf.placeholder(tf.float32, [None, 10])

x_tensor = tf.reshape(x, [-1, img_rows, img_cols, channels])

W_fc_loc1 = weight_variable([7 * 7 * 8, 512])
b_fc_loc1 = bias_variable([512])

W_fc_loc2 = weight_variable([512, 32])
b_fc_loc2 = tf.constant([1.0, 0.0, 0.0, 0.0], dtype=tf.float32)

h_trans = transformer(x_tensor, W_fc_loc1, b_fc_loc1, W_fc_loc2, b_fc_loc2)

W_conv1 = weight_variable([3, 3, 1, 16])
b_conv1 = bias_variable([16])

h_conv1 = tf.nn.relu(tf.nn.conv2d(h_trans, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

W_conv2 = weight_variable([3, 3, 16, 16])
b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

h_conv2_flat = tf.reshape(h_conv2, [-1, 40 * 40 * 16])

W_fc1 = weight_variable([40 * 40 * 16, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(500):
        num_batches = int(X_train.shape[0] / batch_size)
        
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            
            _, loss = sess.run([optimizer, cross_entropy], feed_dict={x: X_train[start:end], y: y_train[start:end], keep_prob: 0.5})
            
            if i % 10 == 0:
                print(f'Epoch {epoch}, Batch {i}, Loss: {loss}')
        
        # Validation
        validation_accuracy = sess.run(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_valid, 1)), tf.float32)), feed_dict={x: X_valid, y: y_valid, keep_prob: 1.0})
        print(f'Epoch {epoch}, Validation Accuracy: {validation_accuracy}')
        
        # Check for NaN values
        if np.isnan(loss):
            print("Loss is NaN, stopping training.")
            break

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)