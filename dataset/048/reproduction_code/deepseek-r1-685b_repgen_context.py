import torch

# Simulate the emission generation process that produces inconsistent tensor sizes
def generate_inconsistent_emissions():
    # First emission chunk (e.g., from first audio segment)
    emission1 = torch.randn(1, 1649, 40)  # (batch, time, features)
    
    # Second emission chunk (e.g., from second audio segment) with different time dimension
    emission2 = torch.randn(1, 1799, 40)  # Inconsistent time dimension
    
    return [emission1, emission2]

# Reproduction of the error
try:
    emissions_arr = generate_inconsistent_emissions()
    # This will fail because time dimensions don't match (1649 vs 1799)
    emissions = torch.cat(emissions_arr, dim=1).squeeze()
except RuntimeError as e:
    print("Error reproduced:")
    print(e)
    print("\nThis occurs when different audio segments produce emissions with")
    print("inconsistent time dimensions during forced alignment.")