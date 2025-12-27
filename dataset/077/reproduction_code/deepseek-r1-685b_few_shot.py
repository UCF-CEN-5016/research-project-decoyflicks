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

def train_model():
    # Simulate problematic data loading
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        'incorrect/path/to/train',  # This path doesn't exist
        image_size=(224, 224),
        batch_size=32
    )
    
    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        'incorrect/path/to/val',  # This path doesn't exist
        image_size=(224, 224),
        batch_size=32
    )
    
    model = create_model()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # This will fail with NotFoundError
    model.fit(
        train_data,
        epochs=5,
        validation_data=val_data
    )
    return model

# This will raise the NotFoundError
try:
    model = train_model()
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")