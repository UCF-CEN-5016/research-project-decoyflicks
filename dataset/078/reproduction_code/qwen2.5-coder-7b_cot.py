import torch
import torch.nn as nn
from vit_pytorch.na_vit_nested_tensor_3d import NaViT as ExternalNaViT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_instance = ExternalNaViT(
    image_size=256,
    patch_size=16,
    dim=512,
    depth=8,
    heads=8,
    mlp_dim=512 * 4,
)

class NaViT(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bind_nest_state_dict = self._bind_nest_state_dict_with_module_info

    @staticmethod
    def _bind_nest_state_dict_with_module_info(model_dict: dict) -> dict:
        """Helper function to bind the nested state dict with module info."""
        new_state = {}
        for k, v in model_dict.items():
            # Add module path information for each key
            if isinstance(k, str):
                module_path = ".".join(k.split(".")[:-1])
                key_part = k.split(".")[-1]
                new_state[f"{module_path}.{key_part}"] = v
        return new_state