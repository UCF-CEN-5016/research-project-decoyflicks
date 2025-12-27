import os
import yaml
import torch
from fairseq import tasks, models, utils

# Set up minimal environment
DATA_ROOT = '/root/autodl-tmp/FormattingData/DATA_ROOT'
CONFIG_YAML = os.path.join(DATA_ROOT, 'config.yaml')
MODEL_PT = '/root/autodl-tmp/xm_transformer_s2ut_en-hk/model.pt'
RESULTS_PATH = '/root/autodl-tmp/results'

# Load YAML configuration file
with open(CONFIG_YAML, 'r') as f:
    config = yaml.safe_load(f)

# Load model
model = models.load_model(MODEL_PT)

# Create speech-to-speech task
task = tasks.get_task('speech_to_speech')

# Create generator
generator = task.build_generator(model, config)

# Define input data
input_data = {'audio': torch.randn(1, 1000)}  # dummy input data

# Triggering conditions: try to access 'input_feat_per_channel' key
try:
    feat_per_channel = config['input_feat_per_channel']
    print(feat_per_channel)
except KeyError:
    print("Key 'input_feat_per_channel' is not in struct")

# Run fairseq-generate command
cmd = f"fairseq-generate {DATA_ROOT} --config-yaml {CONFIG_YAML} --task speech_to_speech --target-is-code --target-code-size 100 --vocoder code_hifigan --path {MODEL_PT} --gen-subset test --max-tokens 50000 --beam 10 --max-len-a 1 --results-path {RESULTS_PATH}"
os.system(cmd)