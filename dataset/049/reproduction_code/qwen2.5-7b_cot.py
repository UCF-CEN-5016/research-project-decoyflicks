import torch

def create_uninitialized_tensor(shape):
    return torch.Tensor(*shape)

def check_tensor_sum(tensor):
    return tensor.sum()

if __name__ == "__main__":
    bias = create_uninitialized_tensor((100, 100))
    print("Sum of uninitialized bias tensor:", check_tensor_sum(bias))

# Example of potential NaN output
# Note: The result may vary due to uninitialized memory