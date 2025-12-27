import torch
from domino.arguments import get_args
from domino.language_model import get_language_model
from domino.modules.enums import AttnMaskType
from domino.modules.module import DominoModule
from domino.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

# Initialize the global cdb object to None to simulate the bug
cdb = None

config = {
    'hidden_size': 4096,
    'padded_vocab_size': 50257
}

args = get_args()
model = DominoModule(config=config)

input_ids = torch.randint(0, 50257, (2, 10))
labels = torch.randint(0, 50257, (2, 10))
position_ids = torch.arange(10).expand(2, -1)
attention_mask = torch.ones((2, 10))

output = model.forward(input_ids, position_ids, attention_mask, labels)

# Attempt to call all_reduce on cdb, which is intentionally uninitialized
try:
    cdb.all_reduce  # This will raise an AttributeError since cdb is None
except AttributeError as e:
    print(e)  # Print the error message to reproduce the bug