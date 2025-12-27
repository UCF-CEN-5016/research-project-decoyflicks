Here is a minimal, self-contained Python script that reproduces the bug:

```
python
import tensorflow as tf
from typing import Dict

def run_experiment() -> (Dict, Dict):
    distribution_strategy = 'some_distribution_strategy'
    task = 'some_task'
    mode = 'train_and_eval'
    params = {'some': 'params'}
    model_dir = 'some_model_dir'
    return {}, {}

if __name__ == '__main__':
    _, _ = run_experiment()
```

