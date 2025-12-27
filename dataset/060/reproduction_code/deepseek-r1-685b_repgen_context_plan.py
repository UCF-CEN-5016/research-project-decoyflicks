import numpy as np
from PIL import Image
from torchvision import transforms

class SegmentationAugmentor:
    def __init__(self, n, m):
        self.n = n
        self.m = m

    def __call__(self, img, mask):
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        img = transforms.functional.apply_transform(img, self.n, self.m)
        mask = transforms.functional.apply_transform(mask, self.n, self.m)

        img = np.array(img)
        mask = np.array(mask)

        return img, mask