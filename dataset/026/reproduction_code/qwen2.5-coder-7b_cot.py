from typing import Any, Sequence, Tuple

def reframe_box_masks_to_image_masks(
    box_masks: Any,
    box_coordinates: Any,
    image_array: Any,
    crop_size: Tuple[int, int],
    extrapolation_value: float = 0.0
) -> Any:
    """Reframe box masks to full-image masks using utils_ops.reframe_box_masks_to_image_masks.

    Preserves the original call semantics: the image dimensions are taken from
    image_array.shape[1] (width) and image_array.shape[2] (height), and crop_size
    is passed as [crop_height, crop_width].

    Args:
        box_masks: Detection masks in box coordinates.
        box_coordinates: Detection boxes corresponding to the masks.
        image_array: Numpy-like array from which width and height are extracted.
        crop_size: Tuple (crop_height, crop_width).
        extrapolation_value: Value used for extrapolation outside mask boundaries.

    Returns:
        Reframed masks mapped to the image coordinates.
    """
    img_width = image_array.shape[1]
    img_height = image_array.shape[2]

    crop_height, crop_width = crop_size

    reframed_masks = utils_ops.reframe_box_masks_to_image_masks(
        box_masks,
        box_coordinates,
        img_width,   # width
        img_height,  # height
        crop_size=[crop_height, crop_width],
        extrapolation_value=extrapolation_value
    )

    return reframed_masks