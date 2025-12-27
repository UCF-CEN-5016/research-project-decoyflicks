import torch

def test_cdb():
    # Attempt to call cdb.all_reduce without initialization
    cdb.all_reduce(torch.randn(2))

test_cdb()