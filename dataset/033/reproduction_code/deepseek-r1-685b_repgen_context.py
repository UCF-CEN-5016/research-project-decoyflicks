import torch
from se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.fiber import Fiber

def create_model():
    fiber_in = Fiber({0: 5, 1: 1})
    fiber_out = Fiber({0: 1})
    fiber_edge = Fiber({0: 4})
    model = SE3Transformer(
        fiber_in=fiber_in,
        fiber_out=fiber_out,
        fiber_edge=fiber_edge,
        num_degrees=2,
        num_channels=4,
        depth=1
    )
    return model

def main():
    model = create_model()

    nodes = {
        '0': torch.randn(10, 5, 1),
        '1': torch.randn(10, 1, 1)
    }
    edges = {
        '0': torch.randn(10, 4, 1)
    }
    edges_idx = torch.randint(0, 10, (2, 20))

    try:
        output = model(nodes, edges, edges_idx)
        print("Success!")
    except RuntimeError as e:
        print(f"Error: {e}")
        print("The error occurs because the attention calculation expects consistent dimensions")
        print("when processing multiple input types. The edge features may need adjustment.")

if __name__ == "__main__":
    main()