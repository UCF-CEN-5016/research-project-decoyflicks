import os
import torch
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

model_dir = "jvn"
if not os.path.exists(model_dir):
    os.system(f"wget https://dl.fbaipublicfiles.com/mms/tts/jvn.tar.gz -O {model_dir}.tar.gz")
    os.system(f"tar -xzf {model_dir}.tar.gz")

models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    model_dir,
    arg_overrides={"vocoder": "hifigan", "fp16": False}
)
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(models, cfg)

text = "こんにちは"
sample = TTSHubInterface.get_model_input(task, text)
wav, rate = TTSHubInterface.get_prediction(task, models[0], generator, sample)