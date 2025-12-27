import matplotlib.pyplot as plt
import numpy as np

def show_batch(images, masks):
    """Display the first image with its corresponding mask from a batch.
    
    Args:
        images: Batch of images
        masks: Corresponding masks
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(images[0])
    axes[0].set_title('Image')
    
    axes[1].imshow(masks[0])
    axes[1].set_title('Mask')
    
    plt.show()

# Test data
test_images = np.random.rand(8, 256, 256, 3)  # Batch of 8 images
test_masks = np.random.randint(0, 2, (8, 256, 256))  # Batch of 8 masks

# Show the first image and mask from the batch
show_batch(test_images, test_masks)