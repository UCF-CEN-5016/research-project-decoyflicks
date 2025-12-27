# ... existing code ...
    def _generate_detections_v2_class_aware(self, predictions, boxes, ...)
        ...
        with tf.name_scope('GenerateDetections'):
            # ... existing code ...

            # Build concatenated_boxes across all classes
            if self._box_list is not None:
                for cls_idx, predicted_boxes in enumerate(self._box_list):
                    # ... processing each class's predicted_boxes ...

                concatenated_boxes = tf.concat([boxes], axis=0)

                # Ensure at least one box exists before concatenation
                if tf.shape(concatenated_boxes)[0] == 0:
                    default_box = tf.constant([[0.0, 0.0, 1.0, 1.0]], dtype=tf.float32)
                    concatenated_boxes = default_box

            else:
                # Default case; should have a single box
                concatenated_boxes = boxes

        return _postprocess(
            detections,
            concatenated_boxes,
            ...)

    # ... rest of the code ...