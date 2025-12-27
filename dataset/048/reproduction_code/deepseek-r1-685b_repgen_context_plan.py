import torch

# Function to generate emissions with consistent time dimension
def generate_consistent_emissions(num_segments, time_dim):
    emissions = []
    for _ in range(num_segments):
        emission = torch.randn(1, time_dim, 40)  # (batch, time, features)
        emissions.append(emission)
    return emissions

# Function to concatenate emissions
def concatenate_emissions(emissions):
    return torch.cat(emissions, dim=1).squeeze()

# Main function to demonstrate the error
def main():
    num_segments = 2
    time_dim = [1649, 1799]

    try:
        emissions_arr = generate_consistent_emissions(num_segments, time_dim[0])
        emissions = concatenate_emissions(emissions_arr)
        print("Concatenated emissions shape:", emissions.shape)
    except RuntimeError as e:
        print("Error reproduced:")
        print(e)
        print("\nThis occurs when different audio segments produce emissions with")
        print("inconsistent time dimensions during forced alignment.")

if __name__ == "__main__":
    main()