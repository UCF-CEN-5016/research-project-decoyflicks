import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Try to enable inline plotting if running in IPython/Jupyter
try:
    from IPython import get_ipython
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("matplotlib", "inline")
except Exception:
    pass


def _extract_images_from_element(element):
    """Extract the image tensor/array from a dataset element.
    Supports tuple/list (e.g., (image, mask)), dicts with 'image' key,
    or plain tensors/arrays.
    """
    if isinstance(element, (list, tuple)):
        return element[0]
    if isinstance(element, dict):
        # Prefer an explicit 'image' key when available
        if "image" in element:
            return element["image"]
        # Fallback to the first value in the dict
        first_value = next(iter(element.values()))
        return first_value
    return element


def show_image_batch(dataset_batch, num_examples=3):
    """Show up to num_examples images from the first batch of a batched dataset."""
    plt.figure(figsize=(10, 10))

    # Get the first batch from the dataset
    iterator = iter(dataset_batch)
    try:
        first_batch = next(iterator)
    except StopIteration:
        return  # Empty dataset, nothing to show

    images = _extract_images_from_element(first_batch)

    # If images is a Tensor or EagerTensor, convert to numpy
    try:
        images_np = images.numpy()
    except Exception:
        # Maybe it's already a numpy array or other sequence
        images_np = images

    # Determine how many examples we can actually show
    try:
        available = images_np.shape[0]
    except Exception:
        # If shape is unavailable, default to 1
        available = 1

    count = min(num_examples, available)

    for i in range(count):
        ax = plt.subplot(1, num_examples, i + 1)
        ax.set_title(f"Image {i + 1}")
        img = images_np[i]
        # If image has channel dimension last and is single-channel, squeeze it
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        ax.imshow(img)
        ax.axis("off")
        plt.grid(False)


# Load a sample dataset (e.g., Pascal VOC for instance segmentation)
raw_dataset, dataset_info = tfds.load(
    "pascal_voc/2018", split="train", with_info=True, shuffle_files=False
)

# Create a batched dataset
batch_size = 32

# Determine a reasonable shuffle buffer size from dataset cardinality when possible
card = tf.data.experimental.cardinality(raw_dataset).numpy()
if card and card > 0:
    shuffle_buffer_size = int(card)
else:
    # Fallback buffer size when cardinality is unknown
    shuffle_buffer_size = 1000

batched_dataset = raw_dataset.shuffle(buffer_size=shuffle_buffer_size).batch(batch_size)

# Show a few images from the first batch
show_image_batch(batched_dataset)