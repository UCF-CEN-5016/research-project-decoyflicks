import tensorflow as tf
from official.vision.ops import augment

def test_randaug_std():
    # Create RandAugment with specific parameters
    ra = augment.RandAugment(
        num_layers=2,
        magnitude=10,
        magnitude_std=0.5,
        prob_to_apply=1.0
    )
    
    # Test image
    image_shape = (224, 224, 3)
    image = tf.zeros(image_shape, dtype=tf.float32)
    
    # Apply augmentation multiple times to observe distribution
    magnitudes = [ra._get_magnitude() for _ in range(1000)]
    
    # The standard deviation should be ~0.5
    mean_magnitude = tf.reduce_mean(magnitudes)
    std_magnitudes = tf.math.reduce_std(magnitudes)
    
    print(f"Mean magnitude: {mean_magnitude:.2f}")
    print(f"Std of magnitudes: {std_magnitudes:.2f}")

if __name__ == "__main__":
    test_randaug_std()