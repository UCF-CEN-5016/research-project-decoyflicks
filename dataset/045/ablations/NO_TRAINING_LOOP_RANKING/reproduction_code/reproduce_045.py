import os
import torch
from fairseq.models.transformer import AugTransformerDecoder

DATA_ROOT = '/root/autodl-tmp/FormattingData/DATA_ROOT'
config_yaml = '/root/autodl-tmp/FormattingData/DATA_ROOT/config.yaml'
model_path = '/root/autodl-tmp/xm_transformer_s2ut_en-hk/model.pt'

os.environ['DATA_ROOT'] = DATA_ROOT

decoder = AugTransformerDecoder(
    args=None,  # Replace with actual args if needed
    dictionary=None,  # Replace with actual dictionary if needed
    embed_tokens=None  # Replace with actual embed_tokens if needed
)

command = f"fairseq-generate {DATA_ROOT} " \
          f"--config-yaml {config_yaml} " \
          f"--task speech_to_speech --target-is-code --target-code-size 100 " \
          f"--vocoder code_hifigan --path {model_path} --gen-subset test " \
          f"--max-tokens 50000 --beam 10 --max-len-a 1 " \
          f"--results-path /root/autodl-tmp/results"

os.system(command)