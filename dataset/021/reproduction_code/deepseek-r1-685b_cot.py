import tensorflow as tf
from official.vision.modeling.layers import detection_generator

def reproduce_bug():
    """Reproduces the ConcatV2 error with empty predictions."""
    # Create empty predictions (0 boxes)
    predicted_boxes = tf.zeros([0, 4])  # Shape [0, 4] means zero boxes
    predicted_scores = tf.zeros([0, 10])  # 10 classes, but zero boxes
    image_info = tf.constant([[512, 512, 1.0, 512, 512, 0.0]])  # Sample image info
    
    # Initialize the detection generator
    generator = detection_generator.MultilevelDetectionGenerator(
        apply_nms=True,
        pre_nms_top_k=5000,
        pre_nms_score_threshold=0.05,
        nms_iou_threshold=0.5,
        max_num_detections=100,
        use_class_agnostic_nms=False
    )
    
    # This will trigger the error
    try:
        _ = generator(predicted_boxes, predicted_scores, image_info)
    except ValueError as e:
        print(f"Error reproduced: {e}")

if __name__ == "__main__":
    reproduce_bug()