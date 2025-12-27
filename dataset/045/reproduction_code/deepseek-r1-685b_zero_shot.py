import torch
from fairseq.models.speech_to_speech.s2ut_transformer import S2UTTransformerModel

model = S2UTTransformerModel.build_model({}, None)
input = {
    "src_tokens": torch.randint(0, 100, (1, 10)),
    "src_lengths": torch.tensor([10]),
    "prev_output_tokens": torch.randint(0, 100, (1, 5))
}
model.forward(**input)