import hydra
from omegaconf import DictConfig

@hydra.main(config_path=".", config_name="duration_predictor.yaml")
def main(cfg: DictConfig):
    if 'input_feat_per_channel' not in cfg:
        raise ValueError("Key 'input_feat_per_channel' is missing from configuration.")
    # Rest of the code