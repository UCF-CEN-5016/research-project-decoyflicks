import torch
import torch.nn.functional as F

def test_gelu_type_error():
    torch.manual_seed(42)
    x = torch.randn(12, 768)
    try:
        F.gelu(x, approximate=True)
    except TypeError as e:
        print(e)
        assert "argument 'approximate' must be str, not bool" in str(e)

test_gelu_type_error()