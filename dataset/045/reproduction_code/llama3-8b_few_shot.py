import fairseq
from pathlib import Path

DATA_ROOT = '/root/autodl-tmp'

# Load model and config
model_path = Path('/root/autodl-tmp/xm_transformer_s2ut_en-hk/model.pt')
config_yaml = '/root/autodl-tmp/FormattingData/DATA_ROOT/config.yaml'
model = fairseq.models.S2UTModel.load_model(model_path)
config = fairseq.utils.parse_config(config_yaml)

# Generate with S2UT
fairseq.generate(
    data_root=DATA_ROOT,
    config_yaml=config_yaml,
    task='speech_to_speech',
    target_is_code=True,
    target_code_size=100,
    vocoder='code_hifigan',
    path=model_path,
    gen_subset='test',
    max_tokens=50000,
    beam=10,
    max_len_a=1,
    results_path='/root/autodl-tmp/results'
)