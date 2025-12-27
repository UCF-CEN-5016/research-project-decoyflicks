import os
import torch
from fairseq import options, tasks, utils
from fairseq.models import BaseFairseqModel

DATA_ROOT = '/root/autodl-tmp/FormattingData'
CONFIG_YAML = '/root/autodl-tmp/FormattingData/DATA_ROOT/config.yaml'
MODEL_PATH = '/root/autodl-tmp/xm_transformer_s2ut_en-hk/model.pt'
RESULTS_PATH = '/root/autodl-tmp/results'

def main():
    os.system(f"fairseq-generate {DATA_ROOT} "
              f"--config-yaml {CONFIG_YAML} "
              f"--task speech_to_speech "
              f"--target-is-code --target-code-size 100 --vocoder code_hifigan "
              f"--path {MODEL_PATH} --gen-subset test "
              f"--max-tokens 50000 --beam 10 --max-len-a 1 "
              f"--results-path {RESULTS_PATH}")

    model = torch.load(MODEL_PATH)
    print("Model state_dict keys:", model.state_dict().keys())
    assert 'input_feat_per_channel' not in model.state_dict(), "Key 'input_feat_per_channel' is missing from the model."

if __name__ == "__main__":
    main()