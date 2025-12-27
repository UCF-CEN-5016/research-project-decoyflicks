import tensorflow as tf

def _generate_detections_v2_class_aware(boxes, scores, pre_nms_top_k=5000):
    if boxes.shape.ndims == 2:
        boxes = tf.expand_dims(boxes, axis=0)
    if scores.shape.ndims == 2:
        scores = tf.expand_dims(scores, axis=0)
    
    selected_boxes_list = []
    for i in range(boxes.shape[0]):
        scores_i = scores[i]
        boxes_i = boxes[i]
        selected_indices = tf.image.non_max_suppression(
            boxes_i, scores_i, pre_nms_top_k, iou_threshold=0.5)
        selected_boxes_list.append(tf.gather(boxes_i, selected_indices))
    
    selected_boxes = tf.concat(selected_boxes_list, axis=0)
    return selected_boxes

boxes = tf.zeros([0, 4], dtype=tf.float32)
scores = tf.zeros([0], dtype=tf.float32)
_generate_detections_v2_class_aware(boxes, scores)