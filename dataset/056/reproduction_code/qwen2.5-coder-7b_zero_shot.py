import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Patches
import matplotlib.pyplot as plt


IMAGE_SIZE = 72
PATCH_SIZE = 8


def generate_random_images(count: int, height: int, width: int, channels: int = 3) -> np.ndarray:
    """Generate random uint8 images."""
    return np.random.randint(0, 255, size=(count, height, width, channels)).astype("uint8")


def display_image(image: np.ndarray, fig_size: tuple = (4, 4)) -> None:
    """Display a single image (numpy uint8)."""
    plt.figure(figsize=fig_size)
    plt.imshow(image.astype("uint8"))
    plt.axis("off")


def create_patches_for_image(image: np.ndarray, image_size: int, patch_size: int) -> tf.Tensor:
    """Resize a single image into a batch and extract patches using Keras Patches layer."""
    # Ensure image is a batch of one and has the requested size
    resized_batch = tf.image.resize(tf.convert_to_tensor([image]), size=(image_size, image_size))
    patches = Patches(patch_size)(resized_batch)
    return patches


def display_patches_grid(patches: tf.Tensor, patch_size: int, fig_size: tuple = (4, 4)) -> None:
    """Display extracted patches in an n x n grid."""
    patches_per_image = int(patches.shape[1])
    grid_n = int(np.sqrt(patches_per_image))

    plt.figure(figsize=fig_size)
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(grid_n, grid_n, i + 1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(tf.convert_to_numpy(patch_img).astype("uint8"))
        plt.axis("off")


def main() -> None:
    images = generate_random_images(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    print(images.shape)

    # Select a random image from the batch (here batch size is 1)
    selected_image = images[np.random.choice(range(images.shape[0]))]

    display_image(selected_image)

    patches = create_patches_for_image(selected_image, IMAGE_SIZE, PATCH_SIZE)

    print(f"Image size: {IMAGE_SIZE} X {IMAGE_SIZE}")
    print(f"Patch size: {PATCH_SIZE} X {PATCH_SIZE}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}")

    display_patches_grid(patches, PATCH_SIZE)


if __name__ == "__main__":
    main()