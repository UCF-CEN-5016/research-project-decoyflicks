import torch

# Simulated emissions and alignments for demonstration purposes
emissions = [torch.randn(100), torch.randn(250)]
alignments = [torch.tensor([0, 1, 0]), torch.tensor([1])]

# Concatenating emissions and alignments
concatenated_emissions = torch.cat(emissions)
concatenated_alignments = torch.cat(alignments)

# Printing shapes to demonstrate mismatch
print("Concatenated Emissions Shape:", concatenated_emissions.shape)
print("Concatenated Alignments Shape:", concatenated_alignments.shape)