import torch
from fairseq import checkpoint_utils
from models.hubert import HubertModel

# Load model and task using provided paths and suffix
hubert, _, _ = checkpoint_utils.load_model_ensemble_and_task(['./assets/hubert/hubert_base.pt'], suffix='')
hubert_model = hubert[0]

# Convert the model to half precision (FP16)
hubert_model = hubert_model.half()

# Define a HuberAdapter class to encapsulate the forward method of the hubert model
class HuberAdapter:
    def __init__(self, model):
        self.model = model

    def forward(self, feats, padding_mask):
        inputs = {'source': feats, 'padding_mask': padding_mask, 'output_layer': 12}
        return self.model.extract_features(**inputs)

# Load input features and padding mask from provided paths
feats = torch.load('./feats.pt')
padding_mask = torch.load('./padding_mask.pt')

# Create a HuberAdapter instance with the loaded model
adapter = HuberAdapter(hubert_model)

# Export the HuberAdapter to ONNX format using torch.onnx.export
torch.onnx.export(adapter, (feats.cuda(), padding_mask.cuda()), 'hubert.onnx', input_names=['feats', 'padding_mask'], output_names=['logits', 'mask'], dynamic_axes={'feats': {0: 'seq'}, 'padding_mask': {0: 'seq'}})