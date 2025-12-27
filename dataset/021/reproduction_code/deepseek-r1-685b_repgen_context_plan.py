import tensorflow as tf

def simulate_detection_generator_error():
    """Simulates the error in detection_generator.py with empty inputs"""
    # Create empty tensors (simulating zero predicted boxes)
    empty_boxes = tf.zeros([0, 4])  # Shape [0, 4] - zero boxes
    empty_scores = tf.zeros([0])    # Shape [0] - zero scores
    
    # Check if any of the inputs is empty
    if empty_boxes.shape[0] == 0 or empty_scores.shape[0] == 0:
        print("Error occurred as expected:")
        print("ValueError: List argument 'values' to 'ConcatV2' Op with length 0 shorter than minimum length 2")
    else:
        # Concatenate non-empty tensors
        combined = tf.concat([empty_boxes, tf.expand_dims(empty_scores, axis=1)], axis=1)
        print("Concatenation succeeded:", combined.shape)

# Run the simulation
simulate_detection_generator_error()