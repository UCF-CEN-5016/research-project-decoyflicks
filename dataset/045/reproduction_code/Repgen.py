import os

DATA_ROOT = '/root/autodl-tmp/FormattingData/DATA_ROOT'

# Create config.yaml
with open(f'{DATA_ROOT}/config.yaml', 'w') as f:
    f.write("""
model: duration_predictor
epochs: 10
dataset:
  train_tsv: ${data_root}/train.tsv
  train_km: ${data_root}/train.km
  valid_tsv: ${data_root}/valid.tsv
  valid_km: ${data_root}/valid.km
  substring: ''
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
""")

# Create train.tsv and train.km
with open(f'{DATA_ROOT}/train.tsv', 'w') as f:
    f.write("1 2 3\n4 5\n6\n7 8 9 10")
with open(f'{DATA_ROOT}/train.km', 'w') as f:
    f.write("1 1 1\n2 2\n3\n4 4 4 4")

# Set environment variable
os.environ['DATA_ROOT'] = DATA_ROOT

# Run the command
import subprocess
subprocess.run(['fairseq-generate', DATA_ROOT, '--config-yaml', f'{DATA_ROOT}/config.yaml', '--task', 'speech_to_speech', '--target-is-code', '--target-code-size', '100', '--vocoder', 'code_hifigan', '--path', '/root/autodl-tmp/xm_transformer_s2ut_en-hk/model.pt', '--gen-subset', 'test', '--max-tokens', '50000', '--beam', '10', '--max-len-a', '1', '--results-path', '/root/autodl-tmp/results'])