import torch
from fairseq.models.hubert import HubertModel, HubertConfig
from fairseq import options, tasks, utils

DATA_ROOT = '/path/to/dataset'

config_yaml = """
task:
  label_rate: 0.1
  # Add other necessary parameters here
"""

with open('/root/autodl-tmp/FormattingData/DATA_ROOT/config.yaml', 'w') as f:
    f.write(config_yaml)

model = HubertModel.build_model(HubertConfig(), tasks.get_task('speech_to_speech'))

fairseq_generate_command = f"""
fairseq-generate {DATA_ROOT} \
--config-yaml /root/autodl-tmp/FormattingData/DATA_ROOT/config.yaml \
--task speech_to_speech --target-is-code --target-code-size 100 --vocoder code_hifigan \
--path /root/autodl-tmp/xm_transformer_s2ut_en-hk/model.pt --gen-subset test \
--max-tokens 50000 --beam 10 --max-len-a 1 \
--results-path /root/autodl-tmp/results
"""

import os
os.system(fairseq_generate_command)