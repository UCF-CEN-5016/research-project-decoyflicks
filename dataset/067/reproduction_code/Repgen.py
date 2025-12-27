import tensorflow as tf
import numpy as np

# Constants
BATCH_SIZE = 8
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CLASSES = 3

# Create a simplified U-Net-like model
def create_model():
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Encoder
    x = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    
    # Decoder
    x = tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding='same')(x)
    
    # Bug: Output has extra dimension that causes issues with SparseCategoricalCrossentropy
    outputs = tf.keras.layers.Conv2D(NUM_CLASSES, 1, padding='same')(x)
    
    return tf.keras.Model(inputs, outputs)

# Create synthetic data
images = np.random.rand(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3).astype(np.float32)

# Bug: Labels have extra dimension [batch, height, width, 1] instead of [batch, height, width]
labels = np.random.randint(0, NUM_CLASSES, 
                         (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 1),  # Extra dimension here
                         dtype=np.int32)

# Create and compile model
model = create_model()

try:
    # This will cause issues due to dimension mismatch
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    print("Model output shape:", model.output_shape)
    print("Label shape:", labels.shape)
    
    # This will train poorly or raise errors due to dimension mismatch
    history = model.fit(
        images,
        labels,
        batch_size=BATCH_SIZE,
        epochs=1
    )
    
except Exception as e:
    print("\nError occurred as expected:")
    print(e)
    
print("\nThe bug occurs because:")
print("1. Model outputs shape:", model.output_shape)
print("2. Labels shape:", labels.shape)
print("3. SparseCategoricalCrossentropy expects labels without the extra dimension")

# Correct implementation would be:
print("\nCorrect implementation:")
correct_labels = labels.reshape(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)
print("Correct label shape:", correct_labels.shape)

# Demonstrate correct usage
try:
    history = model.fit(
        images,
        correct_labels,
        batch_size=BATCH_SIZE,
        epochs=1
    )
    print("\nTraining succeeds with correct label shape")
except Exception as e:
    print("\nUnexpected error with correct implementation:", e)