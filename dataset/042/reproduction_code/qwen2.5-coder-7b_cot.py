import torch
from fairseq import utils
from fairseq.models.hubert import HubertHubert
from huber.hubert import Huber

class BuggyHuberAdapter(torch.nn.Module):
    def __init__(self, model):
        super(BuggyHuberAdapter, self).__init__()
        self.model = model

    def forward(self, feats, padding_mask):
        # Intentionally pass output_layer as a Tensor (not a Python int)
        # to reproduce the AttributeError: 'Tensor' object has no attribute 'is_integer'
        inputs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": torch.tensor(12.0)  # <-- subtle bug trigger
        }
        return self.model.extract_features(**inputs)

def main():
    # Load the pre-trained model (kept as in original script)
    model = HubertHubert.from_pretrained('hubert_base')
    model.half()

    # Prepare input data
    feats = torch.randn(1, 512).half()  # half precision like the model
    padding_mask = torch.zeros(1, 512, dtype=torch.long)
    if torch.cuda.is_available():
        feats = feats.cuda()
        padding_mask = padding_mask.cuda()
        model = model.cuda()

    # Wrap model with Huber then the buggy adapter
    huber_model = Huber(model)
    buggy_adapter = BuggyHuberAdapter(huber_model)

    # Export to ONNX (this should trigger the AttributeError during export)
    dynamic_axes = {
        'feats': {0: 'batch_size', 1: 'sequence_length'},
        'padding_mask': {0: 'batch_size', 1: 'sequence_length'}
    }
    output_names = ['logits', 'mask']
    torch.onnx.export(
        buggy_adapter,
        (feats, padding_mask),
        'hubert_base.onnx',
        input_names=['feats', 'padding_mask'],
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

if __name__ == '__main__':
    main()