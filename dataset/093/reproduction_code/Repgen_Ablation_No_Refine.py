import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load pre-trained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

class MaskedLanguageModel(torch.nn.Module):
    def __init__(self, model):
        super(MaskedLanguageModel, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
        return logits, loss

# Create an instance of the masked language model
masked_lm_model = MaskedLanguageModel(model)

# Example usage
text = "Hello [MASK] world!"
inputs = tokenizer(text, return_tensors="pt")
labels = inputs["input_ids"].clone()
labels[labels == tokenizer.mask_token_id] = -100  # Ignore loss for mask token

# Forward pass
logits, loss = masked_lm_model(**inputs, labels=labels)