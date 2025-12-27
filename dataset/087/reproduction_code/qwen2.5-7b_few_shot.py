import torch
from x_transformers import Attend

# Custom alibi position function that returns 4D tensor
def custom_alibi_pos(pos):
    return torch.randn(2, 3, 4, 5)  # Example 4D shape

def main():
    # Create sample input tensors
    query = torch.randn(2, 3, 4, 5)
    key = torch.randn(2, 3, 4, 5)

    # Initialize the Attend module with parameters
    attention = Attend(
        dim=5,
        alibi_pos=custom_alibi_pos,
        attn_flash=True
    )

    # Compute the output using the Attend module
    output = attention(query, key)

if __name__ == "__main__":
    main()