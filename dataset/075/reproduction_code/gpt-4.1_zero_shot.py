from deepspeed.autotuning.autotuner import Autotuner
from deepspeed.runtime.config import DeepSpeedConfig
import json
import os

ds_config = {
    "train_batch_size": 1,
    "steps_per_print": 1,
    "optimizer": {"type": "Adam", "params": {"lr": 0.001}},
    "zero_optimization": {"stage": 2},
}

os.makedirs("autotuning_exps/profile_model_info", exist_ok=True)
with open("autotuning_exps/profile_model_info/ds_config.json", "w") as f:
    json.dump(ds_config, f)

autotuner = Autotuner("autotuning_exps")
autotuner.tune()