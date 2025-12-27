import tensorflow as tf
from official.projects.edgetpu.vision.modeling import mobilenet_edgetpu_v2_model_blocks

# Assuming _generate_detections_v2_class_aware is defined elsewhere in the codebase
# This function is expected to handle cases where predicted_boxes has length 0 or 1

def test_empty_predicted_boxes():
    batch_size = 1
    height, width, channels = 224, 224, 1
    # Create an empty tensor for predicted boxes
    predicted_boxes = tf.empty((0, 4), dtype=tf.float32)
    # Generate a dummy input tensor
    dummy_input = tf.random.uniform((batch_size, height, width, channels))
    
    # Initialize the model configuration
    model_config = mobilenet_edgetpu_v2_model_blocks.ModelConfig()
    # Get the model output using the dummy input
    model_output = mobilenet_edgetpu_v2_model_blocks.mobilenet_edgetpu_v2(image_input=dummy_input, config=model_config)
    
    try:
        # Call the function that is expected to raise a ValueError for empty predicted boxes
        _generate_detections_v2_class_aware(model_output, predicted_boxes)
    except ValueError as e:
        # Check if the error message matches the expected ValueError
        assert str(e) == "List argument 'values' to 'ConcatV2' Op with length 0 shorter than minimum length 2."

# Run the test function
test_empty_predicted_boxes()