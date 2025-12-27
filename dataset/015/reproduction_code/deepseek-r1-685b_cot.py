def show_batch(image, label, num_of_examples):
    """Demonstrates the bug where num_of_examples is never used.
    
    Args:
        image: Input image tensor
        label: Corresponding label
        num_of_examples: This parameter is defined but never used (the bug)
    """
    # The parameter 'num_of_examples' is passed in but never used in the function
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
    
    # Actual implementation would show/process the batch, but num_of_examples is ignored

# Example usage that would trigger the bug
show_batch(image="dummy_image", label="dummy_label", num_of_examples=5)