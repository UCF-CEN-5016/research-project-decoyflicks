import torch
from fairseq.data import LanguagePairDataset, NoisingDataset, PrependTokenDataset, RoundRobinZipDatasets, SequenceGenerator, TransformEosLangPairDataset

# Load language pair datasets
train_dataset = load_langpair_dataset('en', 'de', data_paths='path_to_train_data', dict_path='path_to_dict')
valid_dataset = load_langpair_dataset('en', 'de', data_paths='path_to_valid_data', dict_path='path_to_dict')

# Combine datasets and create a noising dataset
combined_dataset = RoundRobinZipDatasets(train_dataset, valid_dataset)
noised_dataset = NoisingDataset(combined_dataset, p_noising=0.15)

# Prepend '<mask>' tokens to the dataset
masked_dataset = PrependTokenDataset(noised_dataset, "<mask>")

# Create a SequenceGenerator instance
generator = SequenceGenerator([model], tgt_dict, beam_size=5, stop_words=['<eos>'])

# Transform EOS tokens if necessary
transformed_dataset = TransformEosLangPairDataset(masked_dataset)

# Define the model architecture and extend embeddings
extend_embedding(model, '<mask>', 'en')
add_secial_tokens_to_dict_and_model(dict, '<mask>', model)
extend_embedding(model, '<lang>', 'en')
add_secial_tokens_to_dict_and_model(dict, '<lang>', model)

# Training loop
for epoch in range(num_epochs):
    for batch in transformed_dataset:
        optimizer.zero_grad()
        output = model(**batch)
        loss = compute_loss(output, batch['target'])
        loss.backward()
        optimizer.step()

        if torch.isnan(loss).any():
            print("NaN loss detected in batch")

# Refactored code to fix the identified issues
import torch
from fairseq import checkpoint_utils
import torch

# Load Hubert model from checkpoint
hubert, _, _ = checkpoint_utils.load_model_ensemble_and_task(
    ["./assets/hubert/hubert_base.pt"],
    suffix="",
)
hubert_model = hubert[0]
hubert_model = hubert_model.half()

class HuberAdapter(torch.nn.Module):
    def __init__(self, model):
        super(HuberAdapter, self).__init__()
        self.model = model

    def forward(self, feats, padding_mask):
        inputs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": 12
        }
        return self.model.extract_features(**inputs)

# Load input data
feats = torch.load("./feats.pt")
padding_mask = torch.load("./padding_mask.pt")

# Exporting Adapter into ONNX
adapter = HuberAdapter(hubert_model)
torch.onnx.export(
    adapter,
    (feats.cuda(), padding_mask.cuda()),
    "hubert.onnx",
    input_names=["feats", "padding_mask"],
    output_names=["logits", "mask"],
    dynamic_axes={
        "feats": {0: "seq"},
        "padding_mask": {0: "seq"}
    }
)