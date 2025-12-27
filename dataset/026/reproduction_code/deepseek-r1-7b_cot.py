# After modifications
detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
    detection_masks,
    detection_boxes,
    image_np.shape[1],  # width
    image_np.shape[2],  # height
    crop_size=[image_height, image_width],
    extrapolation_value=0.0)