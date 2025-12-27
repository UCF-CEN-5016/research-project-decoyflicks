import tensorflow as tf
from object_detection.utils import visualization_utils as vis_util

# Load Faster R-CNN model from TensorFlow Hub
model = tf.keras.models.load_model('https://tfhub.dev/tensorflow/faster_rcnn_resnet101_v2_1024x1024/2')

# Load an image for inference (e.g., using PIL)
image_path = 'path/to/image.jpg'
image_np = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)

# Preprocess the image
input_tensor = tf.expand_dims(image_np, 0)
input_tensor = tf.keras.applications.resnet50.preprocess_input(input_tensor)

# Perform inference
detections = model(input_tensor)

# Verify output keys
assert 'detection_classes' in detections.keys()
assert 'detection_scores' in detections.keys()
assert 'detection_boxes' in detections.keys()
assert 'detection_masks' in detections.keys()

# Visualize results
vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    detections['detection_boxes'][0].numpy(),
    detections['detection_classes'][0].numpy().astype(np.int32),
    detections['detection_scores'][0].numpy(),
    category_index=None,
    instance_masks=detections.get('detection_masks_reframed', None),
    use_normalized_coordinates=True,
    line_thickness=8)

# Save or display the image with bounding boxes
cv2.imwrite('output.jpg', image_np)