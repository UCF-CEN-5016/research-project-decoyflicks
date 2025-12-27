import torch
from vit_pytorch import ViT
from vit_pytorch.mpp import MPP

# Configuration constants
IMAGE_SIZE = 256
PATCH_SIZE = 32
NUM_CLASSES = 1000
DIM = 1024
DEPTH = 6
HEADS = 8
MLP_DIM = 2048
DROPOUT = 0.1
EMB_DROPOUT = 0.1

MASK_PROB = 0.15
RANDOM_PATCH_PROB = 0.30
REPLACE_PROB = 0.50

BATCH_SIZE = 20
LR = 3e-4
STEPS = 100
SAVE_PATH = './pretrained-net.pt'


def build_vit() -> ViT:
    """Construct and return a ViT model with the configured hyperparameters."""
    return ViT(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        mlp_dim=MLP_DIM,
        dropout=DROPOUT,
        emb_dropout=EMB_DROPOUT
    )


def build_mpp_trainer(transformer: ViT) -> MPP:
    """Wrap the ViT transformer with the MPP trainer using configured probs."""
    return MPP(
        transformer=transformer,
        patch_size=PATCH_SIZE,
        dim=DIM,
        mask_prob=MASK_PROB,
        random_patch_prob=RANDOM_PATCH_PROB,
        replace_prob=REPLACE_PROB,
    )


def get_optimizer(trainer: MPP) -> torch.optim.Optimizer:
    """Return an Adam optimizer for the trainer parameters."""
    return torch.optim.Adam(trainer.parameters(), lr=LR)


def sample_unlabeled_images(batch_size: int) -> torch.FloatTensor:
    """Simulate sampling a batch of unlabeled images as floats in [0,1]."""
    return torch.FloatTensor(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE).uniform_(0.0, 1.0)


def train(trainer: MPP, optimizer: torch.optim.Optimizer, steps: int, batch_size: int) -> None:
    """Run a simple training loop for a number of steps."""
    for _ in range(steps):
        images = sample_unlabeled_images(batch_size)
        loss = trainer(images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    vit_model = build_vit()
    mpp_trainer = build_mpp_trainer(vit_model)
    optimizer = get_optimizer(mpp_trainer)

    train(mpp_trainer, optimizer, STEPS, BATCH_SIZE)

    # Save the ViT model weights
    torch.save(vit_model.state_dict(), SAVE_PATH)