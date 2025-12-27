import torch
from SE3TransformerPooled import SE3TransformerPooled
from torch.utils.data import TensorDataset, DataLoader

# Define batch size and number of degrees
BATCH_SIZE = 8910
NUM_DEGREES = 5

# Define input dimensions and feature dimensions
input_dim = 8910
node_feat_type_0_dim = 5
node_feat_type_1_dim = 1
output_dim = 1

# Create dummy data for graph features
dummy_graph_feats_type_0 = torch.randn(input_dim, node_feat_type_0_dim)
dummy_graph_feats_type_1 = torch.randn(input_dim, node_feat_type_1_dim)

# Initialize SE3TransformerPooled model
model = SE3TransformerPooled(
    fiber_in=Fiber({0: node_feat_type_0_dim, 1: node_feat_type_1_dim}),
    fiber_out=Fiber({0: NUM_DEGREES * output_dim}),
    fiber_edge=Fiber({0: None}),  # Assuming no edge features
    output_dim=output_dim
)

# Create dummy dataset and DataLoader
dummy_data = {
    '0': dummy_graph_feats_type_0,
    '1': dummy_graph_feats_type_1
}
labels = torch.randint(0, 2, (input_dim,))
dataset = TensorDataset(dummy_data, labels)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Define optimizer and training loop
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
for epoch in range(5):
    for i, (data, targets) in enumerate(dataloader):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()