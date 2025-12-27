import tensorflow as tf
import tensorflow_models as tfm

def reproduce_bug():
    try:
        # Import vision module
        from tensorflow_models import vision
        
        # Verify other modules work (should succeed)
        has_backbones = hasattr(vision, 'backbones')
        has_configs = hasattr(vision, 'configs')
        
        # Try to access augment module (should fail)
        try:
            augment_module = vision.augment
            print("Failed to reproduce bug: augment module exists")
        except AttributeError as e:
            print(f"Bug reproduced: {e}")
            
            # Verify error message matches bug report
            expected_error = "module 'tensorflow_models.vision' has no attribute 'augment'"
            assert expected_error in str(e), "Error message doesn't match bug report"
            
        # Try to use RandAugment (should fail)
        try:
            input_data = tf.random.uniform((32, 224, 224, 3))
            rand_augment = vision.augment.RandAugment()
            augmented_data = rand_augment(input_data)
            print("Failed to reproduce bug: RandAugment usage worked")
        except AttributeError:
            print("Bug reproduced when using RandAugment")
            
    except ImportError as e:
        print(f"Cannot reproduce: vision module not found - {e}")

if __name__ == "__main__":
    reproduce_bug()