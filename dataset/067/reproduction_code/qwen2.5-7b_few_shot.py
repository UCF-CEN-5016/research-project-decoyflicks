import tensorflow as tf

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2, 2, 1)),
        tf.keras.layers.Conv2D(3, (2, 2), activation='softmax')
    ])
    return model

def train_model(model, x, y, loss):
    model.compile(optimizer='adam', loss=loss)
    model.fit(x=x, y=y)

# Define the model
model = create_model()

# Generate one-hot labels (shape: (batch, height, width, num_classes))
labels = tf.constant([[0, 1, 2]])

# Compile with the correct loss for one-hot labels
train_model(model, tf.random.normal([1, 2, 2, 1]), labels, 'categorical_crossentropy')

# Compile with the correct loss for sparse labels
train_model(model, tf.random.normal([1, 2, 2, 1]), labels, 'sparse_categorical_crossentropy')