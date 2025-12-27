import torch.nn as nn

# Modify last convolution layer to match feature map size and add adaptive pooling
discriminator = nn.Sequential(
    # ... existing layers ...
    nn.Conv2d(512, 1, kernel_size=3, padding=1),  # Adjusted for correct reduction
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(1, 1),
)

# In BCE loss:
y_real = Variable(torch.ones(batch_size))
y_fake = Variable(torch.zeros(batch_size))