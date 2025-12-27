import torch
import torch.nn.functional as F

def trigger_gelu_error():
    batch_size = 8
    input_data = torch.randn(batch_size, 12, 768)
    try:
        output = F.gelu(input_data, approximate=True)
    except TypeError as e:
        assert str(e) == "gelu(): argument 'approximate' must be str, not bool"
        print("Caught expected TypeError:", e)

if __name__ == "__main__":
    trigger_gelu_error()