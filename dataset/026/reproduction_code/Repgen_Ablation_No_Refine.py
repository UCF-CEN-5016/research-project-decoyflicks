import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
import numpy as np

# Set up TensorFlow session
sess = tf.Session()

# Load pre-trained Faster R-CNN ResNet V1 model from TensorFlow Hub
model_url = "https://tfhub.dev/google/faster_rcnn_resnet50_coco_2018_07_18/1"
detection_model = tf.saved_model.load(model_url)

# Prepare example images
image_paths = ["path/to/image1.png", "path/to/image2.png"]
images = []
for path in image_paths:
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, dtype=tf.float32) / 255.0
    images.append(image)
images = np.array(images)

# Call visualization function
detection_results = detection_model.signatures['serving_default'](tf.constant(images))
boxes = detection_results['detection_boxes'].numpy()
scores = detection_results['detection_scores'].numpy()
classes = detection_results['detection_classes'].numpy()
masks = detection_results['detection_masks_raw'].numpy()

# Visualize the results
viz_utils.visualize_boxes_and_labels_on_image_array(
    images[0].numpy(),
    boxes[0],
    classes[0].astype(np.int32),
    scores[0],
    category_index={1: {'id': 1, 'name': 'person'}},
    instance_masks=masks[0] if masks is not None else None,
    use_normalized_coordinates=True,
    max_boxes_to_draw=5,
    min_score_thresh=.30,
    agnostic_mode=False,
    line_thickness=2)

# Check for error
try:
    viz_utils.visualize_boxes_and_labels_on_image_array(
        images[0].numpy(),
        boxes[0],
        classes[0].astype(np.int32),
        scores[0],
        category_index={1: {'id': 1, 'name': 'person'}},
        instance_masks=masks[0] if masks is not None else None,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.30,
        agnostic_mode=False,
        line_thickness=2,
        box_ind=0)
except TypeError as e:
    print(e)

# Close TensorFlow session
sess.close()