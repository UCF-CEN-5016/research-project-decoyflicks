import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

def create_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_tensorboard_callback():
    return TensorBoard(log_dir='./logs')

def train_model():
    model = create_model()
    train_data = tf.keras.utils.image_dataset_from_directory(
        'nonexistent_directory/train',
        image_size=(224, 224),
        batch_size=32)
    val_data = tf.keras.utils.image_dataset_from_directory(
        'nonexistent_directory/val',
        image_size=(224, 224),
        batch_size=32)
    tensorboard = create_tensorboard_callback()
    early_stopping = EarlyStopping(patience=3)
    model.fit(x=train_data,
              epochs=10,
              validation_data=val_data,
              callbacks=[tensorboard, early_stopping])
    return model

model = train_model()