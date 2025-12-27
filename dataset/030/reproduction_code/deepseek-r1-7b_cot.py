from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class ModelScheduler:
    def __init__(self, optimizer, scheduler_config):
        super(ModelScheduler, self).__init__()
        self.optimizer = optimizer
        # Ensure the scheduler is a PyTorch LRScheduler subclass
        if isinstance(scheduler_config, dict) and 'scheduler' in scheduler_config:
            self.scheduler = scheduler_config['scheduler'](optimizer, **scheduler_config.get('kwargs', {}))
        else:
            raise ValueError("Custom scheduler must be passed as a dictionary with 'scheduler' key")
    
    def get_last_lr(self):
        return self.scheduler.get_last_lr()
    
    def step(self, epoch=None):
        if not isinstance(epoch, int) and (epoch is None or epoch < 0):
            # This method should only be called with an integer >=0 when using `step()`
            raise TypeError("Argument 'epoch' must be an integer >= 0")
        self.scheduler.step()

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Correct usage within model setup:
model = nnUNet(...)
model.scheduler = ModelScheduler(model.optimizer, {'scheduler': CosineAnnealingWarmRestarts,
                                                  'param_groups': [{'lr': 0.0003}], 
                                                  'T_0': 100, 'T_mult': 2})