import torch

# Function to generate emissions with consistent time dimensions
def generate_consistent_emissions(num_segments, time_dim):
    emissions = []
    for _ in range(num_segments):
        emissions.append(torch.randn(1, time_dim, 40))  # (batch, time, features)
    return emissions

# Function to concatenate emissions along the time dimension
def concatenate_emissions(emissions):
    return torch.cat(emissions, dim=1).squeeze()

# Simulation of the error-free emission generation process
emission_time_dim = 1649
emissions_arr = generate_consistent_emissions(2, emission_time_dim)
emissions = concatenate_emissions(emissions_arr)

print("Successfully concatenated emissions with consistent time dimensions:")
print(emissions)