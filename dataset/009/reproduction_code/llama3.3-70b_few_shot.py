import tensorflow as tf

# Load the pre-trained model
model = tf.saved_model.load('path_to_model/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8')

# Create a DetectionFromImageModule instance
detection_module = model.signatures['serving_default']

# Try to access the 'outputs' attribute, which does not exist
try:
    print(detection_module.outputs)
except AttributeError as e:
    print(f"Error: {e}")

# Instead, you can access the output keys and shapes like this:
print("Output keys:", detection_module.output_keys)
print("Output shapes:", detection_module.output_shapes)