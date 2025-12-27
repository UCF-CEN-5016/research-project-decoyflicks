import torch
from labml.logger import inspect
from labml_nn.transformers.rope import RotaryPositionalEmbeddings

def main():
    x = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=torch.float).reshape(3, 1, 1, 4)
    rotary_pe = RotaryPositionalEmbeddings(3)
    output = rotary_pe(x)
    inspect(output)

if __name__ == '__main__':
    main()