import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# Define constants
NUM_EPOCHS = 100
TRAIN_DIR = 'drive/MyDrive/Dog Vision/train'
VALIDATION_DIR = 'drive/MyDrive/Dog Vision/validation'

# Create data generators
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_data = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

def create_model():
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def create_tensorboard_callback():
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)
    return tensorboard

def create_early_stopping_callback():
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
    return early_stopping

def train_model():
    model = create_model()
    tensorboard = create_tensorboard_callback()
    early_stopping = create_early_stopping_callback()
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, epochs=NUM_EPOCHS, validation_data=val_data, validation_freq=1, callbacks=[tensorboard, early_stopping])
    return model

model = train_model()