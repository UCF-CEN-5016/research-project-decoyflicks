import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def load_image_dataset_from_directory(path):
    return tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(224, 224),
        batch_size=32
    )

def train_model(train_path, val_path):
    train_data = load_image_dataset_from_directory(train_path)
    val_data = load_image_dataset_from_directory(val_path)
    
    model = create_model()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    model.fit(
        train_data,
        epochs=5,
        validation_data=val_data
    )
    
    return model

try:
    model = train_model('correct/path/to/train', 'correct/path/to/val')
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")