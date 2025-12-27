import tensorflow as tf
from tensorflow.keras import layers, models

def create_minimal_efficientdet():
    """Creates a minimal model structure similar to efficientdet's problematic blocks"""
    inputs = tf.keras.Input(shape=(256, 256, 3))
    
    # Simulate stack_6/block_1 structure
    x = layers.Conv2D(64, (1,1), padding='same')(inputs)
    x = layers.BatchNormalization()(x)  # expand_bn
    x = layers.DepthwiseConv2D((3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)  # depthwise_bn
    x = layers.Conv2D(128, (1,1), padding='same')(x)
    x = layers.BatchNormalization()(x)  # project_bn
    
    # Simulate top_bn
    outputs = layers.BatchNormalization()(x)
    
    return models.Model(inputs=inputs, outputs=outputs)

def reproduce_bug():
    # Create model
    model = create_minimal_efficientdet()
    
    # Compile with dummy loss - this will trigger the warning
    model.compile(optimizer='adam')  # Missing loss argument
    
    # Create dummy data
    x_train = tf.random.normal((8, 256, 256, 3))
    y_train = tf.random.normal((8, 256, 256, 128))
    
    # Trigger the warning
    model.fit(x_train, y_train, epochs=1)

if __name__ == '__main__':
    reproduce_bug()