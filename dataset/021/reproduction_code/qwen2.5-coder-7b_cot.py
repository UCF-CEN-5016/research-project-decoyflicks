import tensorflow as tf

# Minimal reproducer of the relevant part of detection_generator.py
class DetectionGenerator:
    def __init__(self, box_list=None):
        # box_list is expected to be a list of tensors (per-class predicted boxes).
        # To reproduce the bug, pass a list of zero-length tensors or an empty list.
        self._box_list = box_list

    def _postprocess(self, detections, concatenated_boxes, *args, **kwargs):
        # Placeholder postprocess; real implementation would convert detections and boxes
        # into final formatted outputs. Here we just return them for simplicity.
        return detections, concatenated_boxes

    def _generate_detections_v2_class_aware(self, predictions, boxes, *args, **kwargs):
        with tf.name_scope('GenerateDetections'):
            # Placeholder for detections computed from predictions.
            detections = predictions  # In real code this would be processed.

            # Build concatenated_boxes across all classes.
            # Intentionally construct a Python list of tensors and call tf.concat on it
            # without ensuring it has at least one (or two) tensors. This will cause
            # a ValueError when the list is empty.
            concat_tensors = []
            if self._box_list is not None:
                for cls_idx, predicted_boxes in enumerate(self._box_list):
                    # Only append if the static shape indicates there are boxes.
                    # If predicted_boxes has a static shape of 0 (zero boxes), we skip appending.
                    # If the static shape is None, we conservatively skip appending as well,
                    # which can lead to an empty concat_tensors list and trigger the bug.
                    if predicted_boxes.shape is not None and predicted_boxes.shape[0] is not None:
                        if int(predicted_boxes.shape[0]) > 0:
                            concat_tensors.append(predicted_boxes)
                    else:
                        # Skip appending when static information is unavailable to intentionally
                        # create the empty-list concat situation in some runtime scenarios.
                        pass

                # Intentionally call tf.concat on concat_tensors even if it might be empty.
                concatenated_boxes = tf.concat(concat_tensors, axis=0)
            else:
                # Default case: use provided single boxes tensor.
                concatenated_boxes = boxes

        return self._postprocess(
            detections,
            concatenated_boxes,
        )


# Example usage that will reproduce the error:
if __name__ == "__main__":
    # Create an instance where per-class predicted boxes are zero-length tensors.
    empty_box = tf.constant([], shape=(0, 4), dtype=tf.float32)
    gen = DetectionGenerator(box_list=[empty_box, empty_box])

    # Dummy predictions and boxes tensors
    predictions = tf.constant([], shape=(0, 6), dtype=tf.float32)
    boxes = tf.constant([], shape=(0, 4), dtype=tf.float32)

    # This call will attempt tf.concat on an empty Python list and raise the reported ValueError.
    try:
        result = gen._generate_detections_v2_class_aware(predictions, boxes)
        print("Result:", result)
    except Exception as e:
        print("Raised exception:", type(e).__name__, e)