import tensorflow as tf
import numpy as np
from object_detection.utils import ops as utils_ops


def generate_fake_detections(num_detections, box_coord_count, mask_size, image_size):
    """
    Simulate detection outputs: bounding boxes, instance masks, and an image array.
    """
    boxes = np.random.rand(num_detections, box_coord_count)
    masks = np.random.rand(num_detections, mask_size[0], mask_size[1])
    image = np.random.rand(image_size[0], image_size[1])
    return boxes, masks, image


def convert_box_masks_to_image_masks(masks, boxes, image_shape):
    """
    Attempt to reframe box masks to image masks using the utility function.
    Preserve original behavior: call the function and catch TypeError, printing the error.
    """
    try:
        # New API for reframe_box_masks_to_image_masks without box_ind
        utils_ops.reframe_box_masks_to_image_masks(
            masks, boxes, image_shape[0], image_shape[1]
        )
    except TypeError as e:
        print("Error:", e)


def main():
    # Simulation parameters
    num_detections = 10
    box_coord_count = 4
    mask_size = (28, 28)
    image_size = (28, 28)

    # Simulate detection outputs
    boxes, masks, image = generate_fake_detections(
        num_detections, box_coord_count, mask_size, image_size
    )

    # Reframe box masks to image masks
    convert_box_masks_to_image_masks(masks, boxes, image.shape)


if __name__ == "__main__":
    main()