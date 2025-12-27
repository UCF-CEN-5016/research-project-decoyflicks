import tensorflow as tf
from cluttered_mnist import load_data, dense_to_one_hot
import numpy as np  # Import numpy to define initial values for b_fc_loc2

batch_size = 100
img_rows, img_cols = 40, 40

# Load MNIST cluttered data
data_path = './data/mnist_sequence1_sample_5distortions5x5.npz'
X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(data_path)

# Convert labels to one-hot encoding
y_train = dense_to_one_hot(y_train, 10)
y_valid = dense_to_one_hot(y_valid, 10)
y_test = dense_to_one_hot(y_test, 10)

# Create a TensorFlow session and initialize variables
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Define placeholders for inputs (x) and labels (y)
    x = tf.placeholder(tf.float32, [None, img_rows*img_cols])
    y = tf.placeholder(tf.float32, [None, 10])

    # Reshape input
    x_reshaped = tf.reshape(x, [-1, img_rows, img_cols, 1])

    # Fully connected layers for the localization network
    W_fc_loc1 = tf.Variable(tf.random_normal([img_rows*img_cols, 20]))
    b_fc_loc1 = tf.Variable(tf.zeros([20]))
    h_fc_loc1 = tf.nn.relu(tf.matmul(x_reshaped, W_fc_loc1) + b_fc_loc1)

    W_fc_loc2 = tf.Variable(tf.random_normal([20, 36]))
    b_fc_loc2 = tf.Variable(tf.constant(0.5, shape=[36]))

    # Set initial values for b_fc_loc2 to identity transformation matrix flattened
    sess.run(b_fc_loc2.assign(np.zeros((36,))))

    # Spatial transformer module (Assuming transformer is a defined function elsewhere)
    theta = transformer(x_reshaped, W_fc_loc1, b_fc_loc1)  # Ensure this function is imported or defined
    output_img = tf.reshape(theta, [-1, 40, 40, 1])

    # Convolutional layers for feature extraction
    conv1 = tf.layers.conv2d(output_img, filters=32, kernel_size=[5, 5], padding='same')
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2])

    # Flatten the output
    flattened = tf.reshape(pool1, [-1, img_rows*img_cols//4])

    # Fully connected layers
    W_fc1 = tf.Variable(tf.random_normal([img_rows*img_cols//4, 1024]))
    b_fc1 = tf.Variable(tf.zeros([1024]))
    h_fc1 = tf.nn.relu(tf.matmul(flattened, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    dropout = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = tf.Variable(tf.random_normal([1024, 10]))
    b_fc2 = tf.Variable(tf.zeros([10]))
    y_pred = tf.matmul(dropout, W_fc2) + b_fc2

    # Loss function
    loss = -tf.reduce_sum(y * tf.log(tf.clip_by_value(y_pred, 1e-10, 1.0)))

    # Optimizer and gradients for localization network
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    grads_and_vars = optimizer.compute_gradients(loss, [b_fc_loc2])

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train the model
    for epoch in range(500):
        for i in range(X_train.shape[0] // batch_size):
            start = i * batch_size
            end = (i + 1) * batch_size
            sess.run(optimizer.apply_gradients(grads_and_vars), feed_dict={x: X_train[start:end], y: y_train[start:end], keep_prob: 0.5})
        if epoch % 10 == 0:
            loss_value = sess.run(loss, feed_dict={x: X_train[:batch_size], y: y_train[:batch_size], keep_prob: 1.0})
            print(f'Epoch {epoch}, Loss: {loss_value}')
            validation_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: y_valid, keep_prob: 1.0})
            print(f'Validation Accuracy: {validation_accuracy}')

    # Monitor GPU memory usage
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())