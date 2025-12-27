import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

# Create a simple model (replace with your actual model)
def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1000, activation='softmax')
    ])
    return model.compile(loss='categorical_crossentropy', optimizer='adam')

# Define TensorBoard callback
def create_tensorboard_callback(log_dir='./graphs'):
    from tensorflow.keras.callbacks import TensorBoard
    tb = TensorBoard(log_dir=log_dir)
    tb.set_model(model)  # Ensure model is set before creating callback
    return tb

def train_model():
    model = create_model()
    
    tensorboard = create_tensorboard_callback()

    # Use valid paths (replace with actual paths to your data)
    base_dir = '/path/to/your/data'
    train_data_dir = os.path.join(base_dir, 'train')
    val_data_dir = os.path.join(base_dir, 'val')

    model.fit(
        x=None,
        epochs=10,
        validation_data=None,
        callbacks=[tensorboard],
        verbose=2
    )
    
    return model

# Try training the model (ensure all data files exist in correct locations)
model = train_model()