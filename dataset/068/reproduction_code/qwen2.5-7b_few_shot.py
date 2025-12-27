import tensorflow as tf
from keras_cv import metrics

# Create fake ground truth and predicted boxes with varying numbers of boxes per image
ground_truth_boxes = tf.constant([
    [[1, 1, 2, 2], [3, 3, 4, 4]],  # Image 0 has 2 boxes
    [[5, 5, 6, 6]],                # Image 1 has 1 box
    [[7, 7, 8, 8], [9, 9, 10, 10]],  # Image 2 has 2 boxes
    [[11, 11, 12, 12]]             # Image 3 has 1 box
])

predicted_boxes = tf.constant([
    [[1, 1, 2, 2]],  # Image 0 has 1 box
    [[5, 5, 6, 6]],   # Image 1 has 1 box
    [[7, 7, 8, 8], [9, 9, 10, 10]],  # Image 2 has 2 boxes
    [[11, 11, 12, 12]]             # Image 3 has 1 box
])

# Initialize COCO metric
coco_metrics = metrics.COCOMetric()

# Update the metric
coco_metrics.update_state(ground_truth_boxes, predicted_boxes)