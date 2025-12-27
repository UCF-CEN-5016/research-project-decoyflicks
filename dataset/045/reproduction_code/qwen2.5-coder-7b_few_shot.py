import os
from pathlib import Path
import yaml

CONFIG_FILE = Path("config.yaml")
FAIRSEQ_MODEL_PATH = "/root/autodl-tmp/xm_transformer_s2ut_en-hk/model.pt"
RESULTS_PATH = "/root/autodl-tmp/results"
DATA_ROOT_VAR = "$DATA_ROOT"


def create_minimal_config():
    return {
        "task": "speech_to_speech",
        "target_is_code": True,
        "target_code_size": 100,
        "vocoder": "code_hifigan",
    }


def write_yaml(path: Path, data: dict):
    with path.open("w") as f:
        yaml.dump(data, f)


def build_fairseq_command(data_root_var: str) -> str:
    parts = [
        "fairseq-generate",
        data_root_var,
        "--config-yaml",
        str(CONFIG_FILE),
        "--task",
        "speech_to_speech",
        "--target-is-code",
        "--target-code-size",
        "100",
        "--vocoder",
        "code_hifigan",
        "--path",
        FAIRSEQ_MODEL_PATH,
        "--gen-subset",
        "test",
        "--max-tokens",
        "50000",
        "--beam",
        "10",
        "--max-len-a",
        "1",
        "--results-path",
        RESULTS_PATH,
    ]
    return " ".join(parts)


def run_command(cmd: str):
    os.system(cmd)


def main():
    config = create_minimal_config()
    write_yaml(CONFIG_FILE, config)
    command = build_fairseq_command(DATA_ROOT_VAR)
    run_command(command)


if __name__ == "__main__":
    main()