import tensorflow as tf

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Dense(10)
    ])
    return model

def load_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def train_model():
    model = create_model()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')

    file_paths = ['drive/MyDrive/Dog Vision/train/17e00d79ad69729522d8705e95939f01.jpg']
    train_data = tf.data.Dataset.from_tensor_slices(file_paths)
    train_data = train_data.map(load_image)
    val_data = train_data

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(train_data, epochs=10, validation_data=val_data, callbacks=[tensorboard_callback])
    
    return model

model = train_model()