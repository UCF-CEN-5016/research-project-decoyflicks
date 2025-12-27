def generate():
    detection_boxes = []
    processed_boxes = []
    if tf.size(boxes) > 0:
        boxes = tf.squeeze(boxes)
        processed_boxes.append(boxes)

    class_aware_processed = []
    if isinstance(class_aware, (list, tuple)):
        for tensor in class_aware:
            if tf.size(tensor) > 0:
                processed_class_aware.append(tf.squeeze(tensor))
    else:
        if tf.size(class_aware) > 0:
            processed_class_aware.append(tf.squeeze(class_aware))

    concat_values = []
    if len(processed_boxes) > 0 and len(processed_class_aware) > 0:
        concat_values.extend([processed_boxes, processed_class_aware])
    elif len(processed_boxes) > 0:
        concat_values.append(processed_boxes[0])
    elif isinstance(class_aware, (list, tuple)) and len(processed_class_aware) > 0:
        if len(processed_class_aware) == 1:
            class aware_tensor = processed_class_aware[0]
            if tf.size(class_aware_tensor) > 0:
                concat_values.append(class_aware_tensor)
    else:
        # Handle cases where neither has elements
        pass

    if len(concat_values) >= 2 and all(tf.size(t) > 0 for t in concat_values):
        concat_value = tf.concat(concat_values, axis=-1)
    else:
        # If fewer than two non-empty tensors, return empty list to avoid errors
        concat_value = []