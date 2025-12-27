import tensorflow as tf
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class ExperimentConfig:
    distribution_strategy: str
    task: str
    mode: str
    params: Dict
    model_dir: str


def build_default_config() -> ExperimentConfig:
    return ExperimentConfig(
        distribution_strategy='some_distribution_strategy',
        task='some_task',
        mode='train_and_eval',
        params={'some': 'params'},
        model_dir='some_model_dir',
    )


def execute_experiment(config: ExperimentConfig) -> Tuple[Dict, Dict]:
    # Core experiment logic placeholder: returns two empty dicts as before
    return {}, {}


def run_experiment() -> Tuple[Dict, Dict]:
    config = build_default_config()
    return execute_experiment(config)


if __name__ == '__main__':
    _, _ = run_experiment()