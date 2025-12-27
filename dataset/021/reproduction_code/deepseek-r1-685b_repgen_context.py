import tensorflow as tf

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
    except tf.errors.InvalidArgumentError as e:
        print("Error occurred as expected:")
        print(f"InvalidArgumentError: {e}")

# Run the simulation
simulate_detection_generator_error()

# Expected output:
# Error occurred as expected:
# InvalidArgumentError: ConcatOp : Dimensions of inputs should match: shape[0] = [0,4] vs. shape[1] = [0,1]