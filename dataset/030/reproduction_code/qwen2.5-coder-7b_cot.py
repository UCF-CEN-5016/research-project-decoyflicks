from typing import Any, Dict, Optional, Callable
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class ModelScheduler:
    """
    A thin wrapper around a torch optimizer learning-rate scheduler.

    Expected scheduler_config format:
      {'scheduler': <callable>, 'kwargs': {...}}

    The 'scheduler' value must be a callable that accepts (optimizer, **kwargs)
    and returns a PyTorch LR scheduler instance.
    """

    def __init__(self, optimizer: Optimizer, config: Dict[str, Any]) -> None:
        self.optimizer = optimizer

        if isinstance(config, dict) and 'scheduler' in config:
            scheduler_ctor: Callable[..., Any] = config['scheduler']
            scheduler_kwargs: Dict[str, Any] = config.get('kwargs', {})
            self._scheduler = scheduler_ctor(optimizer, **scheduler_kwargs)
        else:
            raise ValueError("Custom scheduler must be passed as a dictionary with 'scheduler' key")

    def get_last_lr(self):
        return self._scheduler.get_last_lr()

    def step(self, epoch: Optional[int] = None) -> None:
        # Preserve original validation semantics
        if not isinstance(epoch, int) and (epoch is None or epoch < 0):
            # This method should only be called with an integer >=0 when using `step()`
            raise TypeError("Argument 'epoch' must be an integer >= 0")
        self._scheduler.step()


# Example usage (commented):
# from torch import optim
# optimizer = optim.SGD(model.parameters(), lr=0.1)
# scheduler_wrapper = ModelScheduler(
#     optimizer,
#     {
#         'scheduler': CosineAnnealingWarmRestarts,
#         'kwargs': {'T_0': 100, 'T_mult': 2}
#     }
# )