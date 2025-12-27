tf.image.crop_and_resize(
    box_masks,
    boxes,
    box_ind=tf.range(num_boxes),  # <-- here
    crop_size=[image_height, image_width],
    extrapolation_value=0.0)