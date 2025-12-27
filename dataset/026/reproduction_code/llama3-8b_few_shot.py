import tensorflow as tf
from object_detection.utils import ops

# Load TFHub model and inference script
model = tf.keras.models.load_model('waste_identification_ml/model_inference/TFHub_saved_model_inference')

# Simulate detection output
image_np = tf.random.normal([640, 480, 3])
detection_boxes = tf.random.uniform((10, 4), maxval=640)
detection_masks = tf.random.uniform((10, 480, 640))

try:
    # Try to visualize the results with labels
    detection_masks_reframed = ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes,
        image_np.shape[1], image_np.shape[2],
        box_ind=tf.range(len(detection_boxes))
    )
except TypeError as e:
    print(f"Error: {e}")