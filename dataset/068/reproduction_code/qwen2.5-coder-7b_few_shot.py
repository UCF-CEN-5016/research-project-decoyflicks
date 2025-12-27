import tensorflow as tf
from keras_cv import metrics


def create_sample_boxes():
    """
    Create sample ground truth and predicted bounding boxes for multiple images.
    Returns two TensorFlow constants representing ground truth and predicted boxes.
    """
    ground_truth = tf.constant([
        [[1, 1, 2, 2], [3, 3, 4, 4]],  # Image 0 has 2 boxes
        [[5, 5, 6, 6]],                # Image 1 has 1 box
        [[7, 7, 8, 8], [9, 9, 10, 10]],  # Image 2 has 2 boxes
        [[11, 11, 12, 12]]             # Image 3 has 1 box
    ])

    predicted = tf.constant([
        [[1, 1, 2, 2]],  # Image 0 has 1 box
        [[5, 5, 6, 6]],   # Image 1 has 1 box
        [[7, 7, 8, 8], [9, 9, 10, 10]],  # Image 2 has 2 boxes
        [[11, 11, 12, 12]]             # Image 3 has 1 box
    ])

    return ground_truth, predicted


def evaluate_coco_metric(ground_truth_boxes, predicted_boxes):
    """
    Initialize a COCO metric object and update its state with provided boxes.
    """
    coco_metric = metrics.COCOMetric()
    coco_metric.update_state(ground_truth_boxes, predicted_boxes)
    return coco_metric


def main():
    gt_boxes, pred_boxes = create_sample_boxes()
    _ = evaluate_coco_metric(gt_boxes, pred_boxes)


if __name__ == "__main__":
    main()