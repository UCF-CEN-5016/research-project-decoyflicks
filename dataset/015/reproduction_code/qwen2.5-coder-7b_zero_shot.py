import tensorflow as tf
from typing import Any

def show_batch(dataset: tf.data.Dataset) -> None:
    """No-op placeholder that accepts a batched dataset and does nothing."""
    return None

def create_batched_dataset() -> tf.data.Dataset:
    """Create a simple batched Dataset from a tensor sequence."""
    sample = [1, 2]
    return tf.data.Dataset.from_tensors(sample).batch(1)

def main() -> None:
    ds = create_batched_dataset()
    show_batch(ds)

if __name__ == "__main__":
    main()