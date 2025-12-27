# Reproduction Code for "Key 'input_feat_per_channel' is not in struct" Bug

import os
import torch
from fairseq import distributed_utils, tasks, utils
from fairseq.dataclass import FairseqDataclassConfig
from fairseq.dataclass.utils import convert_config_to_dict

# 1. Set up minimal environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU if available
DATA_ROOT = "/root/autodl-tmp/FormattingData/DATA_ROOT"
CONFIG_PATH = "/root/autodl-tmp/FormattingData/DATA_ROOT/config.yaml"
MODEL_PATH = "/root/autodl-tmp/xm_transformer_s2ut_en-hk/model.pt"
RESULTS_PATH = "/root/autodl-tmp/results"

# 2. Create minimal config.yaml (missing 'input_feat_per_channel')
config_content = """
# Minimal config.yaml that lacks 'input_feat_per_channel'
task:
  type: speech_to_speech
  target_is_code: true
  target_code_size: 100
  vocoder: code_hifigan
"""

# Write config file
with open(CONFIG_PATH, 'w') as f:
    f.write(config_content)

# 3. Trigger the bug by running fairseq-generate with missing config key
try:
    # Initialize distributed environment
    distributed_utils.initalize_distributed_training_environment()
    
    # Load configuration
    config = FairseqDataclassConfig.from_pretrained(CONFIG_PATH)
    config = convert_config_to_dict(config)
    
    # Initialize task
    task = tasks.setup_task(config.task)
    
    # Load model
    model = task.build_model(config.model)
    model.load_pretrained_model(MODEL_PATH)
    
    # This would normally trigger the error when the model is loaded
    # The error occurs because 'input_feat_per_channel' is missing in the config
    
except Exception as e:
    print(f"Error occurred: {str(e)}")