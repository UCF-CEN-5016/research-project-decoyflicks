import os
import deepspeed
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
import argparse

parser = argparse.ArgumentParser()
parser = deepspeed.add_config_arguments(parser)
parser.add_argument('--deepspeed_autotuning', action='store_true')
parser.add_argument('--local_rank', type=int, default=-1)
args = parser.parse_args()

os.environ['LOCAL_RANK'] = str(args.local_rank if args.local_rank != -1 else 0)

deepspeed.init_distributed()

model_config = GPT2Config.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel(model_config)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    args=args,
    dist_init_required=False,
)

if args.deepspeed_autotuning:
    from deepspeed.runtime.autotuning.autotuner import AutoTuner
    autotuner = AutoTuner(model=model, deepspeed_config=args.deepspeed_config)
    autotuner.tune()

input_ids = torch.randint(0, model_config.vocab_size, (1, 16)).cuda()
outputs = model(input_ids)
loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0].mean()
model.backward(loss)
model.step()