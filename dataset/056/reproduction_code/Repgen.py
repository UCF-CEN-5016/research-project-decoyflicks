import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def load_and_preprocess_data():
    """Load and preprocess CIFAR-10 data."""
    (x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255  # Normalize to [0, 1]
    x_train = np.expand_dims(x_train, -1)  # Add channel dimension if needed
    return x_train

class PatchExtractor(layers.Layer):
    """Custom layer to extract image patches."""
    
    def __init__(self, patch_size):
        super(PatchExtractor, self).__init__()
        self.patch_size = patch_size
    
    def call(self, images):
        """Extract patches from input images.
        
        Args:
            images: Tensor of shape (batch, height, width, channels)
            
        Returns:
            Tensor of shape (batch, num_patches, patch_size*patch_size*channels)
        """
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        return tf.reshape(
            patches, 
            [batch_size, -1, self.patch_size * self.patch_size * images.shape[-1]]
        )

def main():
    # Configuration
    IMAGE_SIZE = 72
    PATCH_SIZE = 16
    
    # Load and preprocess data
    x_train = load_and_preprocess_data()
    
    # Convert to tensor and resize
    input_tensor = tf.convert_to_tensor(x_train)
    resized_images = tf.image.resize(
        input_tensor, 
        size=(IMAGE_SIZE, IMAGE_SIZE), 
        method='nearest'
    )
    
    # Initialize patch extractor
    patch_extractor = PatchExtractor(PATCH_SIZE)
    
    # Attempt patch extraction with error handling
    try:
        patches = patch_extractor(resized_images)
        print("Successfully extracted patches!")
        print(f"Patch tensor shape: {patches.shape}")
    except tf.errors.InvalidArgumentError as e:
        print(f"Error during patch extraction: {e.message}")
        print("Note: This implementation expects int32 input but received float32")

if __name__ == "__main__":
    main()