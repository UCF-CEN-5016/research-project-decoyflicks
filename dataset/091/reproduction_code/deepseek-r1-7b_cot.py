import torch
from x_transformers.x_transformers import XTransformer

def generate shortened_sequence():
    model = XTransformer()
    src = torch.randn(258, 512)  # Example encoding sequence
    start_tokens = torch.randn(32, model.config.nemb)
    ENC_SEQ_LEN = 258
    gen_seq_len = 100  # Shorter than ENC_SEQ_LEN
    
    # Create a mask for the original src of length ENC_SEQ_LEN
    mask = torch.tril(torch.ones(gen_seq_len, ENC_seq_len))
    
    sample = model.generate(
        src, 
        start_tokens, 
        ENC_seq_len, 
        mask=mask
    )
    return sample

generate shortened_sequence()

import torch
from x_transformers.x_transformers import XTransformer

def generate_shortened_sequence():
    model = XTransformer()
    
    # Create dummy inputs for encoder and decoder parts
    src = torch.randn(258, 512)
    start_tokens = torch.randn(32, model.config.nemb)
    ENC_seq_len = 258
    gen_seq_len = 100
    
    # Generate the mask dynamically based on decode sequence length
    mask = torch.tril(torch.ones(gen_seq_len, ENC_seq_len))
    
    sample = model.generate(
        src,
        start_tokens,
        ENC_seq_len,
        mask=mask
    )
    
    return sample

generate_shortened_sequence()