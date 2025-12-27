import torch
from fairseq.models import MMSModel
from fairseq.data import Dictionary as MMSDictionary  # Importing MMSDictionary from fairseq

# Sample audio data (16490 samples)
audio_data = torch.randn(1, 16490)

# Load model and dictionary (assuming they are already downloaded)
model = MMSModel.from_pretrained('path_to_model', checkpoint_file='checkpoint_best.pt')
dictionary = MMSDictionary.load('path_to_dictionary')

# Prepare text data
text = "This is a sample text for alignment."
tokens = dictionary.encode_line(text, add_if_not_exist=False, append_eos=True)

# Perform forced alignment
emissions_arr = []  # Simulated emissions array with mismatched shapes
emissions_arr.append(torch.randn(1649))
emissions_arr.append(torch.randn(1799))

# Fixing the array shapes problem by ensuring all tensors have the same size in dimension 1
max_length = max(tensor.size(0) for tensor in emissions_arr)
for i, tensor in enumerate(emissions_arr):
    if tensor.size(0) < max_length:
        emissions_arr[i] = torch.cat((tensor, torch.zeros(max_length - tensor.size(0))), dim=0)

emissions = torch.cat(emissions_arr, dim=1).squeeze()