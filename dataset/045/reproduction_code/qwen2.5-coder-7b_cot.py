import yaml
from typing import Any, Dict, Optional

def load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}

def save_yaml(path: str, data: Dict[str, Any]) -> None:
    with open(path, 'w') as f:
        yaml.dump(data, f)

def ensure_task_section(config: Dict[str, Any]) -> None:
    if 'task' not in config or not isinstance(config['task'], dict):
        config['task'] = {}

def update_speech_to_speech(
    config: Dict[str, Any],
    sample_rate: Optional[int] = None,
    input_feat_per_channel: Optional[int] = None
) -> None:
    ensure_task_section(config)
    sts = config['task'].get('speech_to_speech', {})
    if not isinstance(sts, dict):
        sts = {}
    if sample_rate is not None:
        sts['sample_rate'] = sample_rate
    if input_feat_per_channel is not None:
        sts['input_feat_per_channel'] = input_feat_per_channel
    config['task']['speech_to_speech'] = sts

# Update first configuration file: set only input_feat_per_channel
path1 = '/root/autodl-tmp/FormattingData/DATA_ROOT/config.yaml'
cfg1 = load_yaml(path1)
update_speech_to_speech(cfg1, input_feat_per_channel=128)
save_yaml(path1, cfg1)

# Update local config.yaml: set sample_rate and ensure input_feat_per_channel exists
path2 = 'config.yaml'
cfg2 = load_yaml(path2)
update_speech_to_speech(cfg2, sample_rate=16000, input_feat_per_channel=128)
save_yaml(path2, cfg2)