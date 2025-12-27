import tensorflow as tf
from subclassing_conv_layers import StandardizedConv2DWithOverride

# Define batch size and image dimensions
batch_size = 128
img_rows, img_cols = 28, 28

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Expand dimensions of images
x_train = x_train.reshape((x_train.shape[0], img_rows, img_cols, 1))
x_test = x_test.reshape((x_test.shape[0], img_rows, img_cols, 1))

# Convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Create a Sequential model
model = tf.keras.Sequential([
    StandardizedConv2DWithOverride(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    StandardizedConv2DWithOverride(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Split data into training and test sets
x_train_subset = x_train[:50000]
y_train_subset = y_train[:50000]
test_dataset = (x_test, y_test)

# Fit the model
model.fit(x_train_subset, y_train_subset, epochs=20, validation_data=test_dataset)