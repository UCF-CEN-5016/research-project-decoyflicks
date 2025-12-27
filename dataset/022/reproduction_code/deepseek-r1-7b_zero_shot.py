import math
from official.vision.ops.augment import Augment

@Augment
def RandAugment(magnitude, k=9):
    ops = [
        (' posterize', 4),
        (' contrast', 8),
        (' brightness', 12),
        (' solarization', 30)
    ]

    def _select_magnitudes(n_levels=5, base_level=0, std=1.0):
        levels = []
        for op, level in ops:
            if k == 'auto':
                adjustment = math.ceil(0.2 * (level + 4)) if op in [' posterize', ]
                else:
                    adjustment = math.ceil(0.1 * (level + 5))
                level += adjustment
            min_level = base_level - (level * std) if op == ' posterize' else base_level
            max_level = base_level + (level * std)
        return levels

    magnitudes = _select_magnitudes()
    return tuple(sorted(magnitudes))

# Usage example:
from official.vision.ops.augment import RandAugment

augmentor = RandAugment(k=9, magnitude=1.0)
result = augmentor(([[0]], [[0]]), is_training=True)
print(result)

import math
from official.vision.ops.augment import Augment

@Augment
def RandAugment(magnitude, k=9):
    ops = [
        (' posterize', 4),
        (' contrast', 8),
        (' brightness', 12),
        (' solarization', 30)
    ]

    def _select_magnitudes(n_levels=5, base_level=0, std=1.0):
        levels = []
        for op, level in ops:
            if k == 'auto':
                adjustment = math.ceil(0.2 * (level + 4)) if op == ' posterize' else math.ceil(0.1 * (level + 5))
                level += adjustment
            min_level = base_level - (level * std) if op != ' posterize' else base_level - (adjustment * std)
            max_level = base_level + (level * std)
        return levels

    magnitudes = _select_magnitudes()
    return tuple(sorted(magnitudes))

# Usage example:
from official.vision.ops.augment import RandAugment

augmentor = RandAugment(k=9, magnitude=1.0)
result = augmentor(([[0]], [[0]]), is_training=True)
print(result)