import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# Simulate the environment where the error occurs
def create_model():
    # Using a simple model for reproduction
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model():
    model = create_model()
    
    # Create callbacks (same as original)
    tensorboard = TensorBoard(log_dir='./logs')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    # Simulate the problematic data pipeline
    # This will intentionally create the same NotFoundError
    train_data = tf.data.Dataset.from_tensor_slices({
        'image_path': ['drive/MyDrive/Dog Vision/train/nonexistent_file.jpg'],
        'label': [0]
    }).map(lambda x: (tf.io.read_file(x['image_path']), x['label']))
    
    val_data = train_data.take(1)  # Dummy validation data
    
    # This will trigger the same error
    model.fit(x=train_data,
              epochs=5,
              validation_data=val_data,
              validation_freq=1,
              callbacks=[tensorboard, early_stopping])
    return model

# The error will occur here with the same traceback
try:
    model = train_model()
except Exception as e:
    print("Reproduced error successfully:")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {str(e)}")

import os

def check_files_exist(filepaths):
    for path in filepaths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training cannot proceed - file not found: {path}")

# Use this before model.fit()
check_files_exist(train_filepaths)