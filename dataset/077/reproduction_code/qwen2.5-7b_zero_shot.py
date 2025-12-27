import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Dense(10)
    ])
    return model

def create_tensorboard_callback():
    return tf.keras.callbacks.TensorBoard(log_dir='./logs')

def decode_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    return image

def train_model(train_data):
    model = create_model()
    tensorboard = create_tensorboard_callback()
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_data, epochs=1, callbacks=[tensorboard])
    return model

image_paths = ['drive/MyDrive/Dog Vision/train/17e00d79ad69729522d8705e95939f01.jpg']
train_data = tf.data.Dataset.from_tensor_slices(image_paths)
train_data = train_data.map(decode_image)

model = train_model(train_data)