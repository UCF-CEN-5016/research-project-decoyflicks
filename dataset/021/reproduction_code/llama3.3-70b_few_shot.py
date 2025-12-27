import tensorflow as tf

# Create a function to simulate the '_generate_detections_v2_class_aware' function
def generate_detections_v2_class_aware(predicted_boxes):
    # Check if predicted_boxes is empty
    if len(predicted_boxes) < 2:
        return tf.concat([], axis=0)  # This will raise a ValueError

    # Simulate the ConcatV2 operation
    return tf.concat([tf.constant([1]), tf.constant([2])], axis=0)

# Test the function with an empty list
predicted_boxes = []
try:
    result = generate_detections_v2_class_aware(predicted_boxes)
    print(result)
except ValueError as e:
    print(e)