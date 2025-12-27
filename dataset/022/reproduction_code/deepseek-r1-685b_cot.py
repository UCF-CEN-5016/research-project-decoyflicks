import tensorflow as tf
from official.vision.ops import augment

# Minimal reproduction setup
def test_magnitude_noise_scaling():
    # Create a RandAugment object with specific params
    ra = augment.RandAugment(
        num_layers=2,
        magnitude=10,
        magnitude_std=0.5,  # We want noise with std=0.5
        prob_to_apply=1.0,
    )
    
    # Mock input image
    image = tf.zeros((224, 224, 3), dtype=tf.float32)
    
    # Force deterministic behavior for testing
    tf.random.set_seed(42)
    
    # Apply augmentation
    augmented = ra.distort(image)
    
    # Get the actual noise std being applied
    # (In practice, you'd need to inspect the magnitude values used)
    # This demonstrates the conceptual issue:
    current_implementation_std = 1.0  # Because level_std isn't being multiplied
    expected_std = 0.5
    
    print(f"Expected noise std: {expected_std}")
    print(f"Actual noise std in current implementation: {current_implementation_std}")
    print("The noise std should be scaled by level_std but isn't")

# Run the test
test_magnitude_noise_scaling()