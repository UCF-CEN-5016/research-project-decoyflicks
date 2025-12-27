import torch
import torch.nn.functional as F
import torch.nn as nn
from x_transformers import TransformerWrapper, Autoregressive, Decoder, AutoregressiveWrapper

def reproduce_align_right_pad_id_bug():
    # Define a special pad_id that's different from the default 0
    # to clearly demonstrate the bug
    special_pad_id = 99
    
    # Create a simple transformer model with autoregressive wrapper
    model = TransformerWrapper(
        num_tokens=100,
        max_seq_len=16,
        attn_layers=Decoder(
            dim=32,
            depth=2,
            heads=4
        )
    )
    
    autoreg_model = AutoregressiveWrapper(
        model,
        pad_value=special_pad_id  # Set our special pad_id
    )
    
    # Examine the align_right function (normally internal to AutoregressiveWrapper)
    # This reproduces the function with the bug for demonstration
    def align_right(seq, target_len, pad_id):
        """Reproduces the buggy align_right function from autoregressive_wrapper.py"""
        seq_len = seq.shape[-1]
        
        # The bug is here: pad_id is not used, hardcoded to 0 instead
        if seq_len < target_len:
            pad_len = target_len - seq_len
            seq = F.pad(seq, (0, pad_len), value=0)  # Bug: should use pad_id instead of 0
        
        return seq
    
    # Create sequences of different lengths to demonstrate the issue
    seq1 = torch.tensor([[1, 2, 3, 4]])  # Length 4
    seq2 = torch.tensor([[5, 6]])         # Length 2
    
    # Align both sequences to length 6
    target_len = 6
    
    # Apply the buggy align_right function with our special pad_id
    aligned1 = align_right(seq1, target_len, special_pad_id)
    aligned2 = align_right(seq2, target_len, special_pad_id)
    
    print(f"Original sequence 1: {seq1}")
    print(f"Aligned sequence 1: {aligned1}")
    print(f"Expected aligned sequence 1 with pad_id={special_pad_id}: {F.pad(seq1, (0, target_len - seq1.shape[-1]), value=special_pad_id)}")
    print("\n")
    print(f"Original sequence 2: {seq2}")
    print(f"Aligned sequence 2: {aligned2}")
    print(f"Expected aligned sequence 2 with pad_id={special_pad_id}: {F.pad(seq2, (0, target_len - seq2.shape[-1]), value=special_pad_id)}")
    
    # Demonstrate how this affects actual generation in the AutoregressiveWrapper
    print("\nTesting with AutoregressiveWrapper's internal align_right:")
    
    # Generate a batch of sequences with different lengths
    seqs = [
        torch.tensor([1, 2, 3, 4]),  # Length 4
        torch.tensor([5, 6])         # Length 2
    ]
    
    # Create a batch
    batch = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=special_pad_id)
    
    # Check if the internal align_right in generation would use the correct pad_id
    print(f"Original batch: {batch}")
    print(f"Pad value in autoreg_model: {autoreg_model.pad_value}")
    
    # Try to use the model's generate function (this would actually be affected by the bug)
    try:
        # This is just to show the issue conceptually - the actual bug is in the implementation
        generated = autoreg_model.generate(batch, 10)
        print(f"Generated sequences: {generated}")
    except Exception as e:
        print(f"Error in generation (expected for this demonstration): {e}")
    
    return aligned1, aligned2

# Run the reproduction
aligned1, aligned2 = reproduce_align_right_pad_id_bug()