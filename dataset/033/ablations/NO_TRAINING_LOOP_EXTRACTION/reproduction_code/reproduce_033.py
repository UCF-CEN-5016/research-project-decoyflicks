import torch
import dgl
from SE3Transformer import SE3TransformerPooled
from torch.utils.data import DataLoader
from fiber import Fiber  # Assuming Fiber is defined in a module named fiber

torch.manual_seed(42)
BATCH_SIZE = 32
NODE_FEATURE_DIM = 6

# Load QM9 dataset
qm9_dataset = dgl.data.QM9()
batched_graph = qm9_dataset[0]

# Modify node features
node_feats = {
    '0': batched_graph.ndata['attr'][:, :5, None],
    '1': batched_graph.ndata['attr'][:, 5:6, None]
}
batched_graph.ndata['attr'] = node_feats

# Create DataLoader
data_loader = DataLoader([batched_graph], batch_size=BATCH_SIZE)

# Initialize model
model = SE3TransformerPooled(
    fiber_in=Fiber({0: 5, 1: 1}),
    fiber_out=Fiber({0: 1}),
    fiber_edge=Fiber({0: NODE_FEATURE_DIM}),
    output_dim=1
)

model.train()

# Iterate over DataLoader
for batch in data_loader:
    output = model(batch)
    # Preserve the bug reproduction logic
    assert output.shape != (BATCH_SIZE, 1), "Expected size for first two dimensions of batch2 tensor to be: [8910, 1] but got: [8910, 3]"