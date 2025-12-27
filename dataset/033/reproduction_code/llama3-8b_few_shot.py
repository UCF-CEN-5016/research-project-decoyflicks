import torch
from se3transformer import SE3TransformerPooled

# Custom data module with modified node features
class QM9DataModule(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.NODE_FEATURE_DIM = 6
        self.EDGE_FEATURE_DIM = 1

    def __getitem__(self, index):
        # Simulate batched graph data
        batched_graph = {'attr': torch.randn(8910, self.NODE_FEATURE_DIM)}

        return batched_graph

# Define the SE3Transformer model with modified fiber_in
model = SE3TransformerPooled(
    fiber_in=Fiber({0: 5, 1: 1}),  # original: Fiber({0: datamodule.EDGE_FEATURE_DIM})
    fiber_out=Fiber({0: args.num_degrees * args.num_channels}),
    fiber_edge=Fiber({0: self.EDGE_FEATURE_DIM}),
    output_dim=1,
    tensor_cores=False
)

# Load the dataset and train the model
data_module = QM9DataModule()
dataset = data_module[:]
batched_graphs = [data_module.__getitem__(i) for i in range(len(dataset))]

# Train the model with a single batch
model.train()
for batched_graph in batched_graphs:
    outputs = model(batched_graph)
    loss = torch.nn.functional.mse_loss(outputs, torch.randn_like(outputs))
    loss.backward()
    print(f"Loss: {loss.item()}")

print("Model trained!")