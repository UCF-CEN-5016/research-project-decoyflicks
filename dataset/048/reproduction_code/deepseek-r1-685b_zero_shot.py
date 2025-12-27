import torch
import torchaudio

def generate_emissions():
    emissions_arr = [
        torch.randn(1, 1649, 1024),
        torch.randn(1, 1799, 1024)
    ]
    emissions = torch.cat(emissions_arr, dim=1).squeeze()
    return emissions, 0.02

def get_alignments():
    emissions, stride = generate_emissions()
    return [], stride

def main():
    torch.manual_seed(42)
    get_alignments()

if __name__ == "__main__":
    main()