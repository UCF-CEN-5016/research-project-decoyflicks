import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Minimal environment setup
NUM_EPOCHS = 100
VAL_DATA_DIR = 'drive/MyDrive/Dog Vision/train'

# Triggering conditions - create model and fit it
def train_model():
    model = create_model()
    
    # Set up validation data
    val_data = ImageDataGenerator().flow_from_directory(VAL_DATA_DIR, target_size=(224, 224), batch_size=32)
    
    # Fit the model
    model.fit(x=val_data, epochs=NUM_EPOCHS)
    return model

# Wrap final code in a `try`-`except` block to catch and print error messages
try:
    model = train_model()
except Exception as e:
    print(f"Error: {e}")