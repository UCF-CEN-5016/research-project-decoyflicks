import torch
import dgl
from se3_transformer.model.layers import SE3TransformerPooled, Fiber

torch.manual_seed(42)

batch_size = 32
num_epochs = 10

qm9_dataset = dgl.data.QM9()
data_loader = dgl.dataloading.DataLoader(qm9_dataset, batch_size=batch_size)

for epoch in range(num_epochs):
    for batched_graph in data_loader:
        node_feats = {'0': batched_graph.ndata['attr'][:, :5, None], 
                      '1': batched_graph.ndata['attr'][:, 5:6, None]}
        batched_graph.ndata['attr'] = node_feats['0']  # Set NODE_FEATURE_DIM to 6

        model = SE3TransformerPooled(
            fiber_in=Fiber({0: 5, 1: 1}),
            fiber_out=Fiber({0: 1}),
            fiber_edge=Fiber({0: 1}),
            output_dim=1
        )

        input_tensor = torch.randn(batch_size, 6)
        edge_tensor = torch.randn(batch_size, 1)
        graph = dgl.graph((torch.randint(0, batch_size, (batch_size,)), 
                           torch.randint(0, batch_size, (batch_size,))))

        model.train()
        try:
            output = model(node_feats, {'0': edge_tensor}, graph, {})
        except RuntimeError as e:
            if "Expected size for first two dimensions of batch2 tensor to be: [8910, 1] but got: [8910, 3]" in str(e):
                print("Caught expected RuntimeError:", e)
            else:
                raise e