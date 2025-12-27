import subprocess
import yaml

# Define configuration
config = {
    'task': {
        'type': 'speech_to_speech',
        'target_is_code': True,
        'target_code_size': 100,
        'vocoder': 'code_hifigan'
    }
}

# Save configuration to a YAML file
with open('config.yaml', 'w') as f:
    yaml.safe_dump(config, f)

# Define command arguments
command = [
    'fairseq-generate', 'DATA_ROOT',
    '--config-yaml', 'config.yaml',
    '--task', 'speech_to_speech',
    '--target-is-code',
    '--target-code-size', '100',
    '--vocoder', 'code_hifigan',
    '--path', 'model.pt',
    '--gen-subset', 'test',
    '--max-tokens', '50000',
    '--beam', '10',
    '--max-len-a', '1',
    '--results-path', 'results'
]

# Run the subprocess command
subprocess.run(command, check=False)