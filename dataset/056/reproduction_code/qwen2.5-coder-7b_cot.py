import tensorflow as tf

ops = tf


def resize_to_square_tensor(image, size: int) -> tf.Tensor:
    """Resize an image to a square tensor and cast to float32.

    Args:
        image: Input image (array-like or tensor).
        size: Target width and height (pixels).

    Returns:
        A tf.Tensor containing the resized image cast to float32.
    """
    tensor = ops.convert_to_tensor([image])
    resized = ops.image.resize(tensor, size=(size, size))
    return tf.cast(resized, dtype=tf.float32)


def extract_patches_from_image(image, image_size: int, patch_size):
    """Resize an image and extract patches using the Patches layer.

    Args:
        image: Input image to process.
        image_size: Target square size to which the image will be resized.
        patch_size: Patch size parameter passed to the Patches layer.

    Returns:
        The result of applying Patches(patch_size) to the resized image tensor.
    """
    resized_tensor = resize_to_square_tensor(image, image_size)
    return Patches(patch_size)(resized_tensor)