import torch
import dgl
from SE3Transformer import SE3TransformerPooled, Fiber  # Assuming SE3Transformer is the correct import path

torch.manual_seed(42)
BATCH_SIZE = 32
NODE_FEATURE_DIM = 6

# Load QM9 dataset
# Assuming a function load_qm9 exists that returns a DGL graph
data_loader = load_qm9(batch_size=BATCH_SIZE)  # Ensure load_qm9 is defined elsewhere

# Modify data loading
def modify_data_loading(batched_graph):
    node_feats = {
        '0': batched_graph.ndata['attr'][:, :5, None],  # Node type '0' features
        '1': batched_graph.ndata['attr'][:, 5:6, None]  # Node type '1' features
    }
    return node_feats

# Initialize model
model = SE3TransformerPooled(
    fiber_in=Fiber({0: 5, 1: 1}),  # Input fiber dimensions
    fiber_out=Fiber({0: 1}),        # Output fiber dimensions
    fiber_edge=Fiber({0: NODE_FEATURE_DIM}),  # Edge fiber dimensions
    output_dim=1                    # Output dimension
)

# Create dummy input
dummy_input_0 = torch.randn(BATCH_SIZE, 8910, 5)  # For node type '0'
dummy_input_1 = torch.randn(BATCH_SIZE, 8910, 1)  # For node type '1'

# Forward pass
try:
    output = model(dummy_input_0, dummy_input_1)
except RuntimeError as e:
    print(e)
    assert str(e) == "Expected size for first two dimensions of batch2 tensor to be: [8910, 1] but got: [8910, 3]"

# Training loop to trigger the error
for iteration in range(1):
    try:
        model.train()
        loss = model(dummy_input_0, dummy_input_1)  # This should raise the error
    except RuntimeError as e:
        print(e)
        assert str(e) == "Expected size for first two dimensions of batch2 tensor to be: [8910, 1] but got: [8910, 3]"