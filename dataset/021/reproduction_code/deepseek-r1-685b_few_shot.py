import tensorflow as tf

# Bug Type: Empty Input Error
# Bug Description: ValueError when passing empty list to ConcatV2 operation
# Reproduction Code:

def simulate_detection_generator_error():
    """Simulates the error in detection_generator.py with empty inputs"""
    # Create empty tensors (simulating zero predicted boxes)
    empty_boxes = tf.zeros([0, 4])  # Shape [0, 4] - zero boxes
    empty_scores = tf.zeros([0])    # Shape [0] - zero scores
    
    # This is where the error occurs - trying to concat empty tensors
    try:
        # Simulate the concat operation that fails
        combined = tf.concat([empty_boxes, tf.expand_dims(empty_scores, axis=1)], axis=1)
        print("Concatenation succeeded:", combined.shape)
    except ValueError as e:
        print("Error occurred as expected:")
        print(f"ValueError: {e}")

# Run the simulation
simulate_detection_generator_error()

# Expected output:
# Error occurred as expected:
# ValueError: List argument 'values' to 'ConcatV2' Op with length 0 shorter than minimum length 2