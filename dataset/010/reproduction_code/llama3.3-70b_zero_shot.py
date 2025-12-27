import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the model
def create_model():
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation='relu')(x)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Compile the model
model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create dummy data
train_data = tf.random.normal([10, 256, 256, 3])
train_labels = tf.random.uniform([10, 256, 256, 1], minval=0, maxval=2, dtype=tf.int32)
val_data = tf.random.normal([10, 256, 256, 3])
val_labels = tf.random.uniform([10, 256, 256, 1], minval=0, maxval=2, dtype=tf.int32)

# Train the model
history = model.fit(train_data, train_labels, epochs=1, validation_data=(val_data, val_labels))

# Evaluate the model
loss, accuracy = model.evaluate(val_data, val_labels)
print(f'Validation loss: {loss:.2f}')