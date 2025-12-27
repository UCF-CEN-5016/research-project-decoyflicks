import fairseq
from fairseq.tasks import speech_to_speech

config_yaml = '/root/autodl-tmp/config.yaml'
data_root = '/root/autodl-tmp/DATA_ROOT'

task = speech_to_speech.S2UT(
    input_feat_per_channel=123,  # This line triggers the error
)

fairseq.generate(
    data_root,
    config_yaml=config_yaml,
    task=task,
    max_tokens=50000,
    beam=10,
    max_len_a=1,
)