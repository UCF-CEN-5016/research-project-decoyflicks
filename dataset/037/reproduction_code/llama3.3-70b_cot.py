import torch
import torch.nn.functional as F

# Set up minimal environment
class ValuePE:
    def __init__(self):
        self.rotation_matrix = torch.randn(3, 3)  # Sample rotation matrix

    def rotate(self, value_embedding):
        # Apply rotation to value embedding
        rotated_embedding = torch.matmul(value_embedding, self.rotation_matrix)
        return rotated_embedding

# Create a sample value embedding tensor
value_embedding = torch.randn(10, 3)  # Sample value embedding tensor

# Create an instance of the ValuePE class
value_pe = ValuePE()

# Apply rotation operations to the tensor
rotated_embedding1 = value_pe.rotate(value_embedding)
rotated_embedding2 = value_pe.rotate(rotated_embedding1)

# Print the results
print("Original Value Embedding:")
print(value_embedding)
print("Rotated Embedding (once):")
print(rotated_embedding1)
print("Rotated Embedding (twice):")
print(rotated_embedding2)