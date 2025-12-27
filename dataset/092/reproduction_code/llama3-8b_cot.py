import torch
from transformers import XTModelWrapper

# Set up the Transformer model
model = XTModelWrapper(
    num_tokens=2049,
    max_seq_len=500,
    use_abs_pos_emb=True,
    scaled_sinu_pos_emb=True,
    attn_layers=torch.nn.TransformerDecoderLayer(
        d_model=1024,
        nhead=16,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        cross_attend=True
    )
)

# Generate some random input and context data
i = torch.randint(0, 2048, (2, 20))
context = torch.rand(2, 4, 1024)
context_mask = torch.zeros(2, 4, dtype=torch.bool)

# Set the context mask to all False, indicating padding
context_mask[:, :, :] = True

# Run the model with the generated input and context data
out = model(i, context=context, context_mask=context_mask)

import torch.nn.functional as F

# Define a custom forward pass for our model
class CustomModel(torch.nn.Module):
    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, context=None, context_mask=None):
        # Run the model with the given input and context data
        outputs = self.model(input_ids, attention_mask=attention_mask, context=context, context_mask=context_mask)
        
        # Get the decoder logits
        logits = F.linear(outputs.last_hidden_state, weight=self.model.config.decoder_weight)
        
        return logits

# Create an instance of our custom model
custom_model = CustomModel(model)

# Run the model with the generated input and context data
out = custom_model(i, torch.ones_like(i), context=context, context_mask=context_mask)

if __name__ == "__main__":
    # Run the model with the generated input and context data
    out = custom_model(i, torch.ones_like(i), context=context, context_mask=context_mask)

    # Print out the decoder logits
    print(out)