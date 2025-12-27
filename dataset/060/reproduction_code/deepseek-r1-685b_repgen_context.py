import numpy as np
from PIL import Image
from torchvision import transforms

# Define the RandAugment class
class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment = transforms.Compose([transforms.RandomApply([transforms.AutoAugment()], p=0.5),
                                           transforms.RandomApply([transforms.ColorJitter()], p=0.5)])

    def __call__(self, img):
        for _ in range(self.n):
            img = self.augment(img)
        return img

# Define the function to apply RandAugment on images
def apply_randaugment(img, n, m):
    img = Image.fromarray(img)
    transform = transforms.Compose([transforms.ToPILImage(), RandAugment(n, m), transforms.ToTensor()])
    img = transform(img)
    return img

# Test the code
img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
segmentation_mask = np.random.randint(0, 1, (256, 256), dtype=np.int32)

augmented_img = apply_randaugment(img, n=3, m=5)
augmented_segmentation_mask = apply_randaugment(segmentation_mask, n=3, m=5)

print(augmented_img.shape)
print(augmented_segmentation_mask.shape)