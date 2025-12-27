import os

# Create a minimal config.yaml that is missing the 'input_feat_per_channel' key
config_content = """task: speech_to_speech
target_is_code: True
target_code_size: 100
vocoder: code_hifigan
"""

with open("config.yaml", "w") as f:
    f.write(config_content)

# Run the fairseq-generate command with the incomplete configuration
command = (
    "fairseq-generate $DATA_ROOT "
    "--config-yaml config.yaml "
    "--task speech_to_speech --target-is-code --target-code-size 100 --vocoder code_hifigan "
    "--path /root/autodl-tmp/xm_transformer_s2ut_en-hk/model.pt --gen-subset test "
    "--max-tokens 50000 "
    "--beam 10 --max-len-a 1 "
    "--results-path /root/autodl-tmp/results"
)

os.system(command)