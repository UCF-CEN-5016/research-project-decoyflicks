import fairseq

# Set up minimal environment
data_root = '/root/autodl-tmp/xm_transformer_s2ut_en-hk'
config_yaml_path = '/root/autodl-tmp/FormattingData/DATA_ROOT/config.yaml'

# Add triggering conditions
fairseq.generate(
    data_root,
    config_yaml_path,
    task='speech_to_speech',
    target_is_code=True,
    target_code_size=100,
    vocoder='code_hifigan',
    path='/root/autodl-tmp/xm_transformer_s2ut_en-hk/model.pt',
    gen_subset='test',
    max_tokens=50000,
    beam=10,
    max_len_a=1,
    results_path='/root/autodl-tmp/results'
)