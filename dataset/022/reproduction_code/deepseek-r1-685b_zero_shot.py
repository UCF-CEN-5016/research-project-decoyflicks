import numpy as np

class RandAugment:
    def __init__(self, level_std=0.5):
        self.level_std = level_std

    def _randomly_negate_tensor(self, tensor):
        return tensor * np.random.choice([1.0, -1.0])

    def _get_level_std(self):
        level_std = np.random.normal(loc=1.0, scale=self.level_std)
        return np.clip(level_std, 0.0, 2.0)

    def _apply_autocontrast(self, image, level_std):
        level_std += self._randomly_negate_tensor(self._get_level_std())
        return level_std

def test_randaugment_level_std():
    augment = RandAugment(level_std=0.5)
    results = []
    for _ in range(1000):
        results.append(augment._apply_autocontrast(None, 0))
    print(f"Standard deviation of level_std: {np.std(results):.2f}")

test_randaugment_level_std()