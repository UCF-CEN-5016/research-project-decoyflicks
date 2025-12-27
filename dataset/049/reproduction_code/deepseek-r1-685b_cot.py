import torch

def demonstrate_uninitialized_tensor_issue():
    """Demonstrates the uninitialized tensor problem in bias parameters."""
    # Simulate creating bias parameters like in the original bug
    print("Small tensor (1x1):")
    small_tensor = torch.Tensor(1, 1)
    print(small_tensor.sum())  # Often 0 but technically undefined
    
    print("\nMedium tensor (10x10):")
    medium_tensor = torch.Tensor(10, 10)
    print(medium_tensor.sum())  # Random small value
    
    print("\nLarge tensor (100x100):")
    large_tensor = torch.Tensor(100, 100)
    print(large_tensor.sum())  # Often NaN due to uninitialized memory
    
    # Correct approach using zeros initialization
    print("\nProperly initialized tensor (100x100):")
    proper_tensor = torch.zeros(100, 100)
    print(proper_tensor.sum())  # Always 0 as expected

if __name__ == "__main__":
    demonstrate_uninitialized_tensor_issue()