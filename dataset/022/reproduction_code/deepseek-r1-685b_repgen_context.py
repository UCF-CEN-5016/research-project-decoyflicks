import tensorflow as tf
from official.vision.ops import augment

def test_randaug_std():
    # Create RandAugment with specific params
    ra = augment.RandAugment(
        num_layers=2,
        magnitude=10,
        magnitude_std=0.5,  # Intended to use 0.5 std
        prob_to_apply=1.0
    )
    
    # Test image
    image = tf.zeros((224, 224, 3), dtype=tf.float32)
    
    # Apply augmentation multiple times to observe distribution
    magnitudes = [ra._get_magnitude() for _ in range(1000)]
    
    # The standard deviation should be ~0.5, but will be 1.0 due to the bug
    print(f"Mean magnitude: {tf.reduce_mean(magnitudes):.2f}")
    print(f"Std of magnitudes: {tf.math.reduce_std(magnitudes):.2f}")

test_randaug_std()