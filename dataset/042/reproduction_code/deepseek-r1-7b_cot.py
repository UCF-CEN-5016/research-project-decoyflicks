import torch
from fairseq import utils
from fairseq.models.hubert import HubertHubert
from huber.hubert import Huber

def main():
    # Load the pre-trained model
    model = HubertHubert.from_pretrained('hubert_base')
    
    # Convert to half-precision
    model.half()
    
    # Prepare input data on GPU
    feats = torch.randn(1, 512)  # Example feature tensor
    padding_mask = torch.zeros(1, 512, dtype=torch.long)
    if torch.cuda.is_available():
        feats = feats.cuda()
        padding_mask = padding_mask.cuda()
    
    # Create the HuberAdapter for extraction features
    huber_model = Huber(model)
    huber_adapter = Huber.HubertFeatureExtractor(huber_model)
    
    # Extract features using the model
    extracted_features = huber_adapter.extract_features(feats, padding_mask)
    
    # Define dynamic axes based on input tensors and their shapes
    dynamic_axes = {
        'feats': {0: 'batch_size', 1: 'sequence_length'},
        'padding_mask': {0: 'batch_size', 1: 'sequence_length'}
    }
    
    # Specify all outputs from extract_features()
    output_names = ['extracted_features']
    
    # Export the model to ONNX
    output_path = 'hubert_base.onnx'
    torch.onnx.export(
        huber_adapter,
        (feats, padding_mask),
        output_path,
        input_names=['feats', 'padding_mask'],
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

if __name__ == '__main__':
    main()