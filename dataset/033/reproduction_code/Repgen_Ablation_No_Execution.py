import torch
from SE3Transformer import SE3TransformerPooled, Fiber

batch_size = 10
node_feature_dims = {'0': 5, '1': 1}

# Create random uniform input data with shape (batch_size, num_nodes, feature_dim) for two types of nodes
num_nodes = 8910
inputs = torch.randn(batch_size, num_nodes, sum(node_feature_dims.values()))
node_labels = torch.randint(2, (batch_size, num_nodes), dtype=torch.long)

# Construct the graph with the defined input features
fiber_in = Fiber(node_feature_dims)
fiber_out = Fiber({'0': 3})  # Assuming args.num_degrees and args.num_channels are set to appropriate values
fiber_edge = Fiber({'0': 6})  # Assuming datamodule.EDGE_FEATURE_DIM is set to an appropriate value
model = SE3TransformerPooled(fiber_in=fiber_in, fiber_out=fiber_out, fiber_edge=fiber_edge, output_dim=1)

# Set datamodule.NODE_FEATURE_DIM to 6 for compatibility
datamodule = type('Datamodule', (object,), {'NODE_FEATURE_DIM': 6})

# Define a training loop with one epoch and batch size of 10
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1):
    optimizer.zero_grad()
    out = model(inputs)
    loss = loss_fn(out, node_labels)
    loss.backward()
    optimizer.step()

print("Training completed")