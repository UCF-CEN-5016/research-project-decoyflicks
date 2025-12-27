```
```python
import torch
from torch.nn.functional import log_softmax
from torch.utils.data.dataloader import DataLoader

torch.manual_seed(42)

data = [[["a", "b"], [0, 1]]]
log_probs = torch.tensor([[[0.5, 0.5], [0.3, 0.7]]])

dataloader = DataLoader(data, batch_size=2)
for batch in dataloader:
    log_softmax(batch[1], dim=-1)

log_softmax(log_probs, dim=-1)

