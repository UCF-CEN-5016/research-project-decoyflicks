import tensorflow as tf

def _generate_detections_v2_class_aware(num_classes):
    predicted_boxes = []
    class_indices = []
    
    num_detections = 0
    
    if num_detections == 0:
        return None
    
    detection_scores = tf.constant([])
    
    class_aware_boxes = tf.concat([predicted_boxes, []], axis=0)
    
    return detection_scores, class_aware_boxes

num_classes = 10
detection_scores, class_aware_boxes = _generate_detections_v2_class_aware(num_classes)