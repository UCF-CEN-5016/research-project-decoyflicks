import tensorflow as tf
from typing import Optional, Tuple

def _generate_class_aware_detections(num_classes: int) -> Optional[Tuple[tf.Tensor, tf.Tensor]]:
    """
    Generate class-aware detections.

    Note: This function preserves original behavior. If there are no detections
    (num_detections == 0), it returns None.
    """
    predicted_boxes = []
    class_indices = []

    num_detections = 0

    if num_detections == 0:
        return None

    detection_scores = tf.constant([])
    class_aware_boxes = tf.concat([predicted_boxes, []], axis=0)

    return detection_scores, class_aware_boxes

num_classes = 10
detection_scores, class_aware_boxes = _generate_class_aware_detections(num_classes)